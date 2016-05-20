#include <dirent.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/string_cast.hpp>
#include <jpeglib.h>
#include <time.h>

#define M_PI 3.1415926535897

// interleaved RGB image struct RGB RGB RGB, row major:
// RGBRGBRGB
// RGBRGBRGB
// RGBRGBRGB
// above: example 3 x 3 image.
// 8 bits per channel.
struct Image {
  unsigned char* bytes;
  int width;
  int height;
};

int window_width = 800, window_height = 600;
const std::string window_title = "Procedural Landscape";

const float kNear = 0.0001f;
const float kFar = 1000.0f;
const float kFov = 45.0f;
float aspect = static_cast<float>(window_width) / window_height;

// VBO and VAO descriptors.

// We have these VBOs available for each VAO.
enum {
  kVertexBuffer,
  kIndexBuffer,
  kVertexNormalBuffer,
  kVertexTerrainTypeBuffer,
  kNumVbos
};

// These are our VAOs.
enum {
  kTerrainVao,
  kSkyVao,
  kNumVaos
};

GLuint array_objects[kNumVaos];  // This will store the VAO descriptors.
GLuint buffer_objects[kNumVaos][kNumVbos];  // These will store VBO descriptors.


// 2D Height Map
const int mapSize = 2049;
float map[mapSize][mapSize];

// Height Map Triangles
std::vector<glm::vec4> terrainVerts;
std::vector<glm::uvec3> terrainFaces;

int vertexPresentInSub[mapSize*mapSize];

std::vector<glm::vec2> vertTerrainTypes;

// Sky Data
std::vector<glm::vec4> skyVerts;
std::vector<glm::uvec3> skyFaces;

float last_x = 0.0f, last_y = 0.0f, current_x = 0.0f, current_y = 0.0f;
bool drag_state = false;
int current_button = -1;
float camera_distance = 2.0;
float pan_speed = 0.1f;
float rotation_speed = 0.05f;
float zoom_speed = 0.1f;
glm::vec3 eye = glm::vec3(0.0f, 5.0f, 0.0f);
glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 look = glm::vec3(0.0f, 0.0f, 1.0f);
glm::vec3 tangent = glm::cross(up, look);
glm::vec3 center = eye + camera_distance * look;
glm::mat3 orientation = glm::mat3(tangent, up, look);

glm::mat4 view_matrix = glm::lookAt(eye, center, up);
glm::mat4 projection_matrix =
    glm::perspective((float)(kFov * (M_PI / 180.0f)), aspect, kNear, kFar);
glm::mat4 model_matrix = glm::mat4(1.0f);
glm::mat4 floor_model_matrix = glm::mat4(1.0f);

const char* vertex_shader =
    "#version 330 core\n"
    "uniform vec4 light_position;"
    "in vec4 vertex_position;"
    "out vec4 vs_light_direction;"
    "in vec4 vertex_normal;"
    "in vec2 vertex_terrain_type;"
    "out vec4 v_norm;"
    "out vec2 t_type;"
    "void main() {"
      "gl_Position = vertex_position;"
      "vs_light_direction = vec4(1, 0.5, -1, 0);"
      "v_norm = vertex_normal;"
      "t_type = vertex_terrain_type;"
    "}";

const char* geometry_shader =
    "#version 330 core\n"
    "layout (triangles) in;"
    "layout (triangle_strip, max_vertices = 3) out;"
    "uniform mat4 projection;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform vec4 light_position;"
    "in vec4 vs_light_direction[];"
    "in vec4 v_norm[];"
    "in vec2 t_type[];"
    "out vec4 face_normal;"
    "out vec4 light_direction;"
    "out vec4 world_position;"
    "out vec4 vertex_normal;"
    "out vec2 vertex_terrain_type;"
    "void main() {"
      "int n = 0;"
      "vec3 a = gl_in[0].gl_Position.xyz;"
      "vec3 b = gl_in[1].gl_Position.xyz;"
      "vec3 c = gl_in[2].gl_Position.xyz;"
      "vec3 u = normalize(b - a);"
      "vec3 v = normalize(c - a);"
      "face_normal = normalize(vec4(normalize(cross(u, v)), 0.0));"
      "for (n = 0; n < gl_in.length(); n++) {"
        "light_direction = normalize(vs_light_direction[n]);"
        "world_position = gl_in[n].gl_Position;"
        "gl_Position = projection * view * model * gl_in[n].gl_Position;"
        "vertex_normal = v_norm[n];"
        "vertex_terrain_type = t_type[n];"
        "EmitVertex();"
      "}"
      "EndPrimitive();"
    "}";

const char* floor_fragment_shader =
    "#version 330 core\n"
    "in vec4 face_normal;"
    "in vec4 light_direction;"
    "in vec4 world_position;"
    "in vec4 vertex_normal;"
    "in vec2 vertex_terrain_type;"
    "out vec4 fragment_color;"
    "uniform sampler2D grasstex;"
    "uniform sampler2D mountaintex;"
    "uniform float avg_height;"
    "void main() {"
      "vec2 coord = vec2(world_position.z, world_position.x);"
      "float mtnProb = vertex_terrain_type[0];"
      "fragment_color = mtnProb*texture(mountaintex, coord) + (1 - mtnProb)*texture(grasstex, coord);"
      "float dot_nl = dot(normalize(light_direction), normalize(vertex_normal));"
      "fragment_color[0] *= max(dot_nl, 0);"
      "fragment_color[1] *= max(dot_nl, 0);"
      "fragment_color[2] *= max(dot_nl, 0);"
    "}";

const char* sky_vertex_shader =
    "#version 330 core\n"
    "in vec4 vertex_position;"
    "uniform mat4 projection;"
    "uniform mat4 view;"
    "out vec3 textureCoords;"
    "void main() {"
      "mat4 newview = mat4(mat3(view));"
      "gl_Position = projection * newview * vertex_position;"
      "textureCoords = vec3(vertex_position);"
    "}";

const char* sky_geometry_shader =
    "#version 330 core\n"
    "layout (triangles) in;"
    "layout (triangle_strip, max_vertices = 3) out;"
    "uniform mat4 projection;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform vec4 light_position;"
    "in vec4 vs_light_direction[];"
    "out vec4 face_normal;"
    "out vec4 light_direction;"
    "out vec4 world_position;"
    "void main() {"
      "int n = 0;"
      "vec3 a = gl_in[0].gl_Position.xyz;"
      "vec3 b = gl_in[1].gl_Position.xyz;"
      "vec3 c = gl_in[2].gl_Position.xyz;"
      "vec3 u = normalize(b - a);"
      "vec3 v = normalize(c - a);"
      "face_normal = normalize(vec4(normalize(cross(u, v)), 0.0));"
      "for (n = 0; n < gl_in.length(); n++) {"
        "light_direction = normalize(vs_light_direction[n]);"
        "world_position = gl_in[n].gl_Position;"
        "gl_Position = projection * view * model * gl_in[n].gl_Position;"
      "EmitVertex();"
      "}"
      "EndPrimitive();"
    "}";

const char* sky_fragment_shader =
    "#version 330 core\n"
    "in vec3 textureCoords;"
    "out vec4 fragment_color;"
    "uniform samplerCube tex;"
    "void main() {"
      "fragment_color = texture(tex, textureCoords);"
    "}";

const char* OpenGlErrorToString(GLenum error) {
  switch (error) {
    case GL_NO_ERROR:
      return "GL_NO_ERROR";
      break;
    case GL_INVALID_ENUM:
      return "GL_INVALID_ENUM";
      break;
    case GL_INVALID_VALUE:
      return "GL_INVALID_VALUE";
      break;
    case GL_INVALID_OPERATION:
      return "GL_INVALID_OPERATION";
      break;
    case GL_OUT_OF_MEMORY:
      return "GL_OUT_OF_MEMORY";
      break;
    default:
      return "Unknown Error";
      break;
  }
  return "Unicorns Exist";
}

#define CHECK_SUCCESS(x) \
  if (!(x)) {            \
    glfwTerminate();     \
    exit(EXIT_FAILURE);  \
  }

#define CHECK_GL_SHADER_ERROR(id)                                           \
  {                                                                         \
    GLint status = 0;                                                       \
    GLint length = 0;                                                       \
    glGetShaderiv(id, GL_COMPILE_STATUS, &status);                          \
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);                         \
    if (!status) {                                                          \
      std::string log(length, 0);                                           \
      glGetShaderInfoLog(id, length, nullptr, &log[0]);                     \
      std::cerr << "Line :" << __LINE__ << " OpenGL Shader Error: Log = \n" \
                << &log[0];                                                 \
      glfwTerminate();                                                      \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  }

#define CHECK_GL_PROGRAM_ERROR(id)                                           \
  {                                                                          \
    GLint status = 0;                                                        \
    GLint length = 0;                                                        \
    glGetProgramiv(id, GL_LINK_STATUS, &status);                             \
    glGetProgramiv(id, GL_INFO_LOG_LENGTH, &length);                         \
    if (!status) {                                                           \
      std::string log(length, 0);                                            \
      glGetProgramInfoLog(id, length, nullptr, &log[0]);                     \
      std::cerr << "Line :" << __LINE__ << " OpenGL Program Error: Log = \n" \
                << &log[0];                                                  \
      glfwTerminate();                                                       \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  }

#define CHECK_GL_ERROR(statement)                                             \
  {                                                                           \
    { statement; }                                                            \
    GLenum error = GL_NO_ERROR;                                               \
    if ((error = glGetError()) != GL_NO_ERROR) {                              \
      std::cerr << "Line :" << __LINE__ << " OpenGL Error: code  = " << error \
                << " description =  " << OpenGlErrorToString(error);          \
      glfwTerminate();                                                        \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  size_t count = std::min(v.size(), static_cast<size_t>(10));
  for (size_t i = 0; i < count; ++i) os << i << " " << v[i] << "\n";
  os << "size = " << v.size() << "\n";
  return os;
}

void LoadObj(const std::string& file, std::vector<glm::vec4>& vertices,
             std::vector<glm::uvec3>& indices) {
  std::ifstream in(file);
  int i = 0, j = 0;
  glm::vec4 vertex = glm::vec4(0.0, 0.0, 0.0, 1.0);
  glm::uvec3 face_indices = glm::uvec3(0, 0, 0);
  while (in.good()) {
    char c = in.get();
    switch (c) {
      case 'v':
        in >> vertex[0] >> vertex[1] >> vertex[2];
        vertices.push_back(vertex);
        break;
      case 'f':
        in >> face_indices[0] >> face_indices[1] >> face_indices[2];
        face_indices -= 1;
        indices.push_back(face_indices);
        break;
      default:
        break;
    }
  }
  in.close();
}

void SaveJPEG(const std::string& filename, int image_width, int image_height,
              const unsigned char* pixels) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE* outfile;
  JSAMPROW row_pointer[1];
  int row_stride;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  CHECK_SUCCESS((outfile = fopen(filename.c_str(), "wb")) != NULL)

  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = image_width;
  cinfo.image_height = image_height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 100, true);
  jpeg_start_compress(&cinfo, true);

  row_stride = image_width * 3;

  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = const_cast<unsigned char*>(
        &pixels[(cinfo.image_height - 1 - cinfo.next_scanline) * row_stride]);
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);

  jpeg_destroy_compress(&cinfo);
}

bool LoadJPEG(const std::string& file_name, Image* image) {
  FILE* file = fopen(file_name.c_str(), "rb");
  struct jpeg_decompress_struct info;
  struct jpeg_error_mgr err;

  info.err = jpeg_std_error(&err);
  jpeg_create_decompress(&info);

  CHECK_SUCCESS(file != NULL);

  jpeg_stdio_src(&info, file);
  jpeg_read_header(&info, true);
  jpeg_start_decompress(&info);

  image->width = info.output_width;
  image->height = info.output_height;

  int channels = info.num_components;
  long size = image->width * image->height * 3;

  image->bytes = new unsigned char[size];

  int a = (channels > 2 ? 1 : 0);
  int b = (channels > 2 ? 2 : 0);
  std::vector<unsigned char> scan_line(image->width * channels, 0);
  unsigned char* p1 = &scan_line[0];
  unsigned char** p2 = &p1;
  unsigned char* out_scan_line = &image->bytes[0];
  while (info.output_scanline < info.output_height) {
    jpeg_read_scanlines(&info, p2, 1);
    for (int i = 0; i < image->width; ++i) {
      out_scan_line[3 * i] = scan_line[channels * i];
      out_scan_line[3 * i + 1] = scan_line[channels * i + a];
      out_scan_line[3 * i + 2] = scan_line[channels * i + b];
    }
    out_scan_line += image->width * 3;
  }
  jpeg_finish_decompress(&info);
  fclose(file);
  return true;
}

void ErrorCallback(int error, const char* description) {
  std::cerr << "GLFW Error: " << description << "\n";
}

bool keys[1024];

void KeyCallback(GLFWwindow* window, int key, int scancode, int action,
                 int mods) {
  std::cout << "here" << std::endl;
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
  if(action == GLFW_PRESS)
    keys[key] = true;
  else if(action == GLFW_RELEASE && key != GLFW_KEY_SPACE)
    keys[key] = false;
  else if (key == GLFW_KEY_J && action != GLFW_RELEASE) {
    std::vector<unsigned char> pixels(3 * window_width * window_height, 0);
    CHECK_GL_ERROR(glReadPixels(0, 0, window_width, window_height, GL_RGB,
                                GL_UNSIGNED_BYTE, &pixels[0]));
    std::string filename = "capture.jpg";
    std::cout << "Encoding and saving to file '" + filename + "'\n";
    SaveJPEG(filename, window_width, window_height, &pixels[0]);
  }
}

float t = 0.0f, v = 0.15f;
glm::vec3 startPosition;
float maxCameraDisplacementBeforeMovingRenderWindow = 20.0f;
bool renderWindow = true;
float renderRadius = 50.0f;

void moveCamera() {
  if (keys[GLFW_KEY_W]) {
    glm::vec3 look_xz = glm::normalize(glm::vec3(look[0], 0.0f, look[2]));
    eye -= zoom_speed * look_xz;
  } if (keys[GLFW_KEY_S]) {
    glm::vec3 look_xz = glm::normalize(glm::vec3(look[0], 0.0f, look[2]));
    eye += zoom_speed * look_xz;
  } if (keys[GLFW_KEY_A]) {
    glm::vec3 tangent_xz = glm::normalize(glm::vec3(tangent[0], 0.0f, tangent[2]));
    eye -= pan_speed * tangent_xz;
  } if (keys[GLFW_KEY_D]) {
    glm::vec3 tangent_xz = glm::normalize(glm::vec3(tangent[0], 0.0f, tangent[2]));
    eye += pan_speed * tangent_xz;
  }

  int row = (int)(4*(eye[2] + float(mapSize)/8));
  int col = (int)(4*(eye[0] + float(mapSize)/8));

  if (keys[GLFW_KEY_W] || keys[GLFW_KEY_S] || keys[GLFW_KEY_A] || keys[GLFW_KEY_D]) {
    glm::vec3 displacementVector = eye - startPosition;
    if (!renderWindow && glm::length(displacementVector) > maxCameraDisplacementBeforeMovingRenderWindow) {
      renderWindow = true;
      startPosition = eye;
    }
  }

  if (!keys[GLFW_KEY_SPACE] && (keys[GLFW_KEY_W] || keys[GLFW_KEY_S] || keys[GLFW_KEY_A] || keys[GLFW_KEY_D])) {
    // Move eye up and down based on height map
    eye[1] = map[row][col] + 0.2f;
  }

  // jump physics
  if (keys[GLFW_KEY_SPACE]) {
    t += 0.005f;
    eye[1] += v*t;
    v -= 0.1*t;
    if(eye[1] < map[row][col] + 0.2f) {
      eye[1] = map[row][col] + 0.2f;
      t = 0.0f;
      v = 0.15f;
      keys[GLFW_KEY_SPACE] = false;
    }
  }
}

void MousePosCallback(GLFWwindow* window, double mouse_x, double mouse_y) {
  last_x = current_x;
  last_y = current_y;
  current_x = mouse_x;
  current_y = window_height - mouse_y;
  float delta_x = current_x - last_x;
  float delta_y = current_y - last_y;
  if (sqrt(delta_x * delta_x + delta_y * delta_y) < 1e-15) return;
  glm::vec3 mouse_direction = glm::normalize(glm::vec3(delta_x, delta_y, 0.0f));
  glm::vec2 mouse_start = glm::vec2(last_x, last_y);
  glm::vec2 mouse_end = glm::vec2(current_x, current_y);
  glm::uvec4 viewport = glm::uvec4(0, 0, window_width, window_height);
  if (drag_state && current_button == GLFW_MOUSE_BUTTON_LEFT) {
    glm::vec3 oup = glm::column(orientation, 1);
    orientation[1] = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 axis = glm::normalize(
        orientation * glm::vec3(mouse_direction.y, -mouse_direction.x, 0.0f));
    orientation =
        glm::mat3(glm::rotate(rotation_speed, axis) * glm::mat4(orientation));
    orientation[1] = oup;

    tangent = glm::column(orientation, 0);
    up = glm::column(orientation, 1);
    look = glm::column(orientation, 2);
  }
}

void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  drag_state = (action == GLFW_PRESS);
  current_button = button;
}

int randRangeInt(int min, int max) {
  return std::rand() % (max - min) + min;
}

double randRange(float min, float max) {
  return ((double)std::rand()/(double)RAND_MAX) * (max - min) + min;
}

float smoothness;
float avgFractalY = 0, avgPerlinY = 0;

void CHIterative(std::pair<int,int> c1, std::pair<int,int> c2, std::pair<int,int> c3, std::pair<int,int> c4, int w, float range) {
  int s = 1;
  while (w >= 1) {
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) {
        std::pair<int,int> cOne = std::make_pair(c1.first + w*i, c1.second + w*j);
        std::pair<int,int> cTwo = std::make_pair(cOne.first, cOne.second + w);
        std::pair<int,int> cThree = std::make_pair(cOne.first + w, cOne.second + w);
        std::pair<int,int> cFour = std::make_pair(cOne.first + w, cOne.second);

        std::pair<int,int> mid = std::make_pair(cOne.first + w/2, cOne.second + w/2);
        float avg = map[cOne.first][cOne.second] + map[cTwo.first][cTwo.second] + map[cThree.first][cThree.second] + map[cFour.first][cFour.second];
        avg /= 4;
        map[mid.first][mid.second] = avg + randRange(-range/2, range/2);
      }
    }

    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) {
        std::pair<int,int> cOne = std::make_pair(c1.first + w*i, c1.second + w*j);
        std::pair<int,int> cTwo = std::make_pair(cOne.first, cOne.second + w);
        std::pair<int,int> cThree = std::make_pair(cOne.first + w, cOne.second + w);
        std::pair<int,int> cFour = std::make_pair(cOne.first + w, cOne.second);

        std::pair<int,int> mid = std::make_pair(cOne.first + w/2, cOne.second + w/2);

        std::pair<int,int> e1 = std::make_pair(cOne.first, mid.second);
        float avg = map[cOne.first][cOne.second] + map[cTwo.first][cTwo.second] + map[mid.first][mid.second];
        if (e1.first - w/2 >= 0) {
          avg += map[e1.first - w/2][mid.second];
          avg /= 4;
        } else {
          //avg += map[mapSize + (e1.first - w/2)][mid.second];
          avg /= 3;
        }
        map[e1.first][e1.second] = avg + randRange(-range/2, range/2);

        std::pair<int,int> e2 = std::make_pair(mid.first, cTwo.second);
        avg = map[cTwo.first][cTwo.second] + map[cThree.first][cThree.second] + map[mid.first][mid.second];
        if (e2.second + w/2 < mapSize) {
          avg += map[e2.first][e2.second + w/2];
          avg /= 4;
        } else {
          //avg += map[e2.first][(e2.second + w/2) - mapSize];
          avg /= 3;
        }
        map[e2.first][e2.second] = avg + randRange(-range/2, range/2);

        std::pair<int,int> e3 = std::make_pair(cThree.first, mid.second);
        avg = map[cThree.first][cThree.second] + map[cFour.first][cFour.second] + map[mid.first][mid.second];
        if (e3.first + w/2 < mapSize) {
          avg += map[e3.first + w/2][e3.second];
          avg /= 4;
        } else {
          //avg += map[(e3.first + w/2) - mapSize][e3.second];
          avg /= 3;
        }
        map[e3.first][e3.second] = avg + randRange(-range/2, range/2);

        std::pair<int,int> e4 = std::make_pair(mid.first, cFour.second);
        avg = map[cFour.first][cFour.second] + map[cOne.first][cOne.second] + map[mid.first][mid.second];
        if (e4.second - w/2 >= 0) {
          avg += map[e4.first][e4.second - w/2];
          avg /= 4;
        } else {
          //avg += map[e4.first][mapSize + (e4.second - w/2)];
          avg /=3;
        }
        map[e4.first][e4.second] = avg + randRange(-range/2, range/2);
      }
    }

    s *= 2;
    w /= 2;
    range = std::pow(2.0, -smoothness)*range;
  }
}

std::vector<int> P;

float fade(float x) {
  return std::pow(x, 3) * (x * (x * 6 - 15) + 10);
}

float lerp(float x, float a, float b) {
  return a + x * (b-a);
}

float grad(int h, float x, float y, float z) {
  h = h & 15;
  float u = y;
  float v = z;
  if (h < 8) u = x;
  if (h < 4) {
    v = y;
  } else if (h == 12 || h == 14) {
    v = x;
  }
  return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}


// Taken from Perlin's personal reference
float PerlinNoise(float x, float y) {
  float z = 0.5f;

  int X = (int) std::floor(x) & 255;
  int Y = (int) std::floor(y) & 255;
  int Z = (int) std::floor(z) & 255;

  x -= std::floor(x);
  y -= std::floor(y);
  z -= std::floor(z);

  float u = fade(x);
  float v = fade(y);
  float w = fade(z);

  int A = P[X] + Y;
  int B = P[X + 1] + Y;

  int AA = P[A] + Z;
  int BA = P[B] + Z;

  int AB = P[A + 1] + Z;
  int BB = P[B + 1] + Z;

  return lerp(w, lerp(v, lerp(u, grad(P[AA], x, y, z), grad(P[BA], x-1, y, z)), lerp(u, grad(P[AB], x, y-1, z), grad(P[BB], x-1, y-1, z))),
                  lerp(v, lerp(u, grad(P[AA + 1], x, y, z-1), grad(P[BA+1], x-1, y, z-1)), lerp(u, grad(P[AB+1], x, y-1, z-1), grad(P[BB+1], x-1, y-1, z-1))));
}

// diamond-square algorithm
void computeHeightsFractal(float s, float r) {
  smoothness = s;
  map[0][0] = map[0][mapSize-1] = map[mapSize-1][mapSize-1] = map[mapSize-1][0] = ((double)std::rand()/(double)RAND_MAX) * 4 - 2;
  CHIterative(std::make_pair(0, 0), std::make_pair(0, mapSize-1), std::make_pair(mapSize-1, mapSize-1), std::make_pair(mapSize-1, 0), mapSize - 1, r);
}

// perlin noise
void computeHeightsPerlin(float a, float b, float c, float f1x, float f1y, float f2x, float f2y, float f3x, float f3y) {
  float lowest = 1.0f;
  float highest = 0.0f;
  for (int x = 0; x < mapSize; x++) {
    for (int y = 0; y < mapSize; y++) {
      float nx = ((float)x)/513 - 0.5;//((float) mapSize) - 0.5;
      float ny = ((float)y)/513 - 0.5;//((float) mapSize) - 0.5;
      map[x][y] = a*PerlinNoise(f1x*nx, f1y*ny) + b*PerlinNoise(f2x*nx, f2y*ny) + c*PerlinNoise(f3x*nx, f3y*ny);

      if (map[x][y] > highest) highest = map[x][y];
      if (map[x][y] < lowest) lowest = map[x][y];

      map[x][y] = std::pow(map[x][y], 4);
    }
  }
}

float mapFractal[mapSize][mapSize];
float mapPerlin[mapSize][mapSize];

// mix of fractals and perlin noise
void computeHeights() {
  // compute fractal array, with avg y value
  computeHeightsFractal(0.6, 16);
  for(int i=0; i < mapSize; i++) {
    for(int j=0; j < mapSize; j++) {
      avgFractalY += map[i][j];
      mapFractal[i][j] = map[i][j];
      map[i][j] = -1;
    }
  }
  avgFractalY /= mapSize*mapSize;

  // compute perlin array, with avg y value
  computeHeightsPerlin(1.0, 0.5, 0.25, 10, 10, 50, 50, 40, 40);
  for(int i=0; i < mapSize; i++) {
    for(int j=0; j < mapSize; j++) {
      avgPerlinY += map[i][j];
      mapPerlin[i][j] = map[i][j];
      map[i][j] = -1;
    }
  }
  avgPerlinY /= mapSize*mapSize;

  // move perlin plane up to fractal plane
  for(int i=0; i < mapSize; i++)
    for(int j=0; j < mapSize; j++)
      mapPerlin[i][j] += (avgFractalY - avgPerlinY);

  for(int i=0; i < mapSize; i++) {
    for(int j=0; j < mapSize; j++) {
      if(mapFractal[i][j] > avgFractalY) {
        map[i][j] = mapFractal[i][j];
        vertTerrainTypes.push_back(glm::vec2(1.0f, 1.0f));
      } else {
        map[i][j] = mapPerlin[i][j];
        vertTerrainTypes.push_back(glm::vec2(0.0f, 0.0f));
      }
    }
  }
}

void generateTriangles() {
  for(int i = 0; i < mapSize; i++) {
    for(int j = 0; j < mapSize; j++) {
      terrainVerts.push_back(glm::vec4(-float(mapSize)/8 + j/4.0, (map[i][j]==-1)?0:map[i][j], -float(mapSize)/8 + i/4.0, 1.0f));
    }
  }
  for(int i = 0; i < mapSize - 1; i++) {
    for(int j = 0; j < mapSize - 1; j++) {
      terrainFaces.push_back(glm::uvec3(i*mapSize + j, (i+1)*mapSize + j,  i*mapSize + (j+1)));
      terrainFaces.push_back(glm::uvec3((i+1)*mapSize + (j+1), i*mapSize + (j+1), (i+1)*mapSize + j));
    }
  }
}

GLuint generateSky(std::vector<Image>& images) {
  // Populate the vertices skybox
  skyVerts.push_back(glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f));
  skyVerts.push_back(glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f));
  skyVerts.push_back(glm::vec4(-1.0f, -1.0f, 1.0f, 1.0f));
  skyVerts.push_back(glm::vec4(-1.0f, 1.0f, 1.0f, 1.0f));
  skyVerts.push_back(glm::vec4(1.0f, -1.0f, -1.0f, 1.0f));
  skyVerts.push_back(glm::vec4(1.0f, 1.0f, -1.0f, 1.0f));
  skyVerts.push_back(glm::vec4(1.0f, -1.0f, 1.0f, 1.0f));
  skyVerts.push_back(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));

  skyFaces.push_back(glm::uvec3(2, 0, 1));
  skyFaces.push_back(glm::uvec3(3, 2, 1));

  skyFaces.push_back(glm::uvec3(6, 2, 3));
  skyFaces.push_back(glm::uvec3(7, 6, 3));

  skyFaces.push_back(glm::uvec3(4, 6, 7));
  skyFaces.push_back(glm::uvec3(5, 4, 7));

  skyFaces.push_back(glm::uvec3(4, 5, 1));
  skyFaces.push_back(glm::uvec3(4, 1, 0));

  skyFaces.push_back(glm::uvec3(4, 0, 2));
  skyFaces.push_back(glm::uvec3(4, 2, 6));

  skyFaces.push_back(glm::uvec3(5, 7, 3));
  skyFaces.push_back(glm::uvec3(3, 1, 5));

  // Load the texture into opengl
  GLuint textureID;
  glGenTextures(1, &textureID);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

  int texw = 1024;
  int texh = 1024;
  glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB, texw, texh, 0, GL_RGB, GL_UNSIGNED_BYTE, images[4].bytes);
  glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB, texw, texh, 0, GL_RGB, GL_UNSIGNED_BYTE, images[1].bytes);
  glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB, texw, texh, 0, GL_RGB, GL_UNSIGNED_BYTE, images[2].bytes);
  glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB, texw, texh, 0, GL_RGB, GL_UNSIGNED_BYTE, images[0].bytes);
  glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB, texw, texh, 0, GL_RGB, GL_UNSIGNED_BYTE, images[0].bytes);
  glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB, texw, texh, 0, GL_RGB, GL_UNSIGNED_BYTE, images[3].bytes);

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

  return textureID;
}

int main(int argc, char* argv[]) {
  if (!glfwInit()) exit(EXIT_FAILURE);
  glfwSetErrorCallback(ErrorCallback);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);
  GLFWwindow* window = glfwCreateWindow(window_width, window_height,
                                        &window_title[0], nullptr, nullptr);
  CHECK_SUCCESS(window != nullptr);

  glfwMakeContextCurrent(window);
  glewExperimental = GL_TRUE;
  CHECK_SUCCESS(glewInit() == GLEW_OK);
  glGetError();  // clear GLEW's error for it

  glfwSetKeyCallback(window, KeyCallback);
  glfwSetCursorPosCallback(window, MousePosCallback);
  glfwSetMouseButtonCallback(window, MouseButtonCallback);
  glfwSwapInterval(1);
  const GLubyte* renderer = glGetString(GL_RENDERER);  // get renderer string
  const GLubyte* version = glGetString(GL_VERSION);    // version as a string
  std::cout << "Renderer: " << renderer << "\n";
  std::cout << "OpenGL version supported:" << version << "\n";

  // Set up Perlin Noise Permutation
  for (int i = 0; i < 256; i++) {
    P.push_back(i);
  }
  std::random_shuffle(P.begin(), P.end());
  for (int i = 0; i < 256; i++) {
    P.push_back(P[i]);
  }

  // Initialize 2D Height Map to -1
  for (int i = 0; i < mapSize; i++) {
    for (int j = 0; j < mapSize; j++) {
      map[i][j] = -1;
    }
  }

  std::srand((unsigned)time(NULL));

  // Create random terrain
  computeHeights();

  // Put terrain into vertices and faces form
  generateTriangles();

  // Generate per-vertex normals
  std::vector<glm::vec4> vertex_normals(terrainVerts.size());
  std::vector<int> count(terrainVerts.size());
  for(int i=0; i < terrainFaces.size(); i++) {
    glm::uvec3 face = terrainFaces[i];
    glm::vec3 normal = glm::normalize(glm::cross(glm::vec3( terrainVerts[face[1]] - terrainVerts[face[0]]),
                                                    glm::vec3( terrainVerts[face[2]] - terrainVerts[face[0]])));
    glm::vec4 n = glm::vec4(normal, 0.0f);
    vertex_normals[face[0]] += n;
    vertex_normals[face[1]] += n;
    vertex_normals[face[2]] += n;

    count[face[0]]++;
    count[face[1]]++;
    count[face[2]]++;
  }
  for (int i=0; i < terrainVerts.size(); i++) {
    vertex_normals[i] /= count[i];
  }

  // Set eye on top of the plane and update everything else
  eye = glm::vec3(0.0f, map[mapSize/2][mapSize/2] + 0.2f, 0.0f);
  up = glm::vec3(0.0f, 1.0f, 0.0f);
  look = glm::vec3(0.0f, 0.0f, 1.0f);
  tangent = glm::cross(up, look);
  center = eye + camera_distance * look;
  orientation = glm::mat3(tangent, up, look);

  view_matrix = glm::lookAt(eye, center, up);
  projection_matrix =
    glm::perspective((float)(kFov * (M_PI / 180.0f)), aspect, kNear, kFar);
  startPosition = eye;

  std::vector<std::string> jpeg_file_names;
  DIR* dir;
  struct dirent* entry;
  CHECK_SUCCESS((dir = opendir("./textures")) != NULL);
  while ((entry = readdir(dir)) != NULL) {
    std::string file_name(entry->d_name);
    std::transform(file_name.begin(), file_name.end(), file_name.begin(),
                   tolower);
    if (file_name.find(".jpg") != std::string::npos) {
      jpeg_file_names.push_back(file_name);
    }
  }
  closedir(dir);

  std::sort(jpeg_file_names.rbegin(), jpeg_file_names.rend());
  std::vector<Image> images(jpeg_file_names.size());
  for (int i = 0; i < jpeg_file_names.size(); ++i) {
    std::string file_name = "./textures/" + jpeg_file_names[i];
    LoadJPEG(file_name, &images[i]);
    std::cout << "Loaded '" << file_name << "' width = " << images[i].width
              << " height = " << images[i].height << "\n";
  }

  // Generate sky texture
  GLuint textureid = generateSky(images);

  // Set up the grass texture in a slot
  glActiveTexture(GL_TEXTURE1);
  GLuint grasstex;
  CHECK_GL_ERROR(glGenTextures(1, &grasstex));
  GLuint sampler;
  CHECK_GL_ERROR(glGenSamplers(1, &sampler));

  CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, grasstex));
  CHECK_GL_ERROR(glTexStorage2D(GL_TEXTURE_2D,  1, GL_RGB8, images[5].width, images[5].height));
  CHECK_GL_ERROR(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, images[5].width, images[5].height, GL_RGB, GL_UNSIGNED_BYTE, (const GLvoid*) images[5].bytes));

  glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  glBindTexture(GL_TEXTURE_2D, grasstex);
  glBindSampler(1, sampler);

  // Set up the mountain texture in a slot
  glActiveTexture(GL_TEXTURE2);
  GLuint mountaintex;
  CHECK_GL_ERROR(glGenTextures(1, &mountaintex));
  GLuint samplerm;
  CHECK_GL_ERROR(glGenSamplers(1, &samplerm));

  CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, mountaintex));
  CHECK_GL_ERROR(glTexStorage2D(GL_TEXTURE_2D,  1, GL_RGB8, images[6].width, images[6].height));
  CHECK_GL_ERROR(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, images[6].width, images[6].height, GL_RGB, GL_UNSIGNED_BYTE, (const GLvoid*) images[6].bytes));

  glSamplerParameteri(samplerm, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glSamplerParameteri(samplerm, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  glBindTexture(GL_TEXTURE_2D, mountaintex);
  glBindSampler(2, samplerm);

  //////////////////////////////////////////////////////////////////////////////

  // Setup our VAOs.
  CHECK_GL_ERROR(glGenVertexArrays(kNumVaos, array_objects));

  // Setup the object array object.

  // Switch to the Terrain VAO.
  CHECK_GL_ERROR(glBindVertexArray(array_objects[kTerrainVao]));

  // Generate buffer objects
  CHECK_GL_ERROR(glGenBuffers(kNumVbos, &buffer_objects[kTerrainVao][0]));

  // Setup vertex data in a VBO
  CHECK_GL_ERROR(
      glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kTerrainVao][kVertexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(float) * (3.2 * renderRadius * renderRadius) * 4,
                              nullptr, GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
  CHECK_GL_ERROR(glEnableVertexAttribArray(0));

  // Setup element array buffer.
  CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                              buffer_objects[kTerrainVao][kIndexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                              sizeof(uint32_t) * (3.2 * renderRadius * renderRadius) * 3,
                              nullptr, GL_STATIC_DRAW));

  // Setup vertex normal data in a VBO
  CHECK_GL_ERROR(
      glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kTerrainVao][kVertexNormalBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(float) * (3 * 3.2 * renderRadius * renderRadius) * 4,
                              nullptr, GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0));
  CHECK_GL_ERROR(glEnableVertexAttribArray(1));

  // Setup vertex terrain type data in a VBO
  CHECK_GL_ERROR(
      glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kTerrainVao][kVertexTerrainTypeBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(float) * (3.2 * renderRadius * renderRadius) * 2,
                              nullptr, GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0));
  CHECK_GL_ERROR(glEnableVertexAttribArray(2));

  // Triangle shaders

  // Setup vertex shader.
  GLuint vertex_shader_id = 0;
  const char* vertex_source_pointer = vertex_shader;
  CHECK_GL_ERROR(vertex_shader_id = glCreateShader(GL_VERTEX_SHADER));
  CHECK_GL_ERROR(
      glShaderSource(vertex_shader_id, 1, &vertex_source_pointer, nullptr));
  glCompileShader(vertex_shader_id);
  CHECK_GL_SHADER_ERROR(vertex_shader_id);

  // Setup geometry shader.
  GLuint geometry_shader_id = 0;
  const char* geometry_source_pointer = geometry_shader;
  CHECK_GL_ERROR(geometry_shader_id = glCreateShader(GL_GEOMETRY_SHADER));
  CHECK_GL_ERROR(
      glShaderSource(geometry_shader_id, 1, &geometry_source_pointer, nullptr));
  glCompileShader(geometry_shader_id);
  CHECK_GL_SHADER_ERROR(geometry_shader_id);

  // Setup floor fragment shader.
  GLuint floor_fragment_shader_id = 0;
  const char* floor_fragment_source_pointer = floor_fragment_shader;
  CHECK_GL_ERROR(floor_fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER));
  CHECK_GL_ERROR(glShaderSource(floor_fragment_shader_id, 1,
                                &floor_fragment_source_pointer, nullptr));
  glCompileShader(floor_fragment_shader_id);
  CHECK_GL_SHADER_ERROR(floor_fragment_shader_id);

  // Let's create our floor program.
  GLuint terrain_program_id = 0;
  CHECK_GL_ERROR(terrain_program_id = glCreateProgram());
  CHECK_GL_ERROR(glAttachShader(terrain_program_id, vertex_shader_id));
  CHECK_GL_ERROR(glAttachShader(terrain_program_id, floor_fragment_shader_id));
  CHECK_GL_ERROR(glAttachShader(terrain_program_id, geometry_shader_id));

  // Bind attributes.
  CHECK_GL_ERROR(glBindAttribLocation(terrain_program_id, 0, "vertex_position"));
  CHECK_GL_ERROR(glBindAttribLocation(terrain_program_id, 1, "vertex_normal"));
  CHECK_GL_ERROR(glBindAttribLocation(terrain_program_id, 2, "vertex_terrain_type"));
  CHECK_GL_ERROR(glBindFragDataLocation(terrain_program_id, 0, "fragment_color"));
  glLinkProgram(terrain_program_id);
  CHECK_GL_PROGRAM_ERROR(terrain_program_id);

  // Get the uniform locations.
  GLint terrain_projection_matrix_location = 0;
  CHECK_GL_ERROR(terrain_projection_matrix_location =
                     glGetUniformLocation(terrain_program_id, "projection"));
  GLint terrain_model_matrix_location = 0;
  CHECK_GL_ERROR(terrain_model_matrix_location =
                     glGetUniformLocation(terrain_program_id, "model"));
  GLint terrain_view_matrix_location = 0;
  CHECK_GL_ERROR(terrain_view_matrix_location =
                     glGetUniformLocation(terrain_program_id, "view"));
  GLint terrain_light_position_location = 0;
  CHECK_GL_ERROR(terrain_light_position_location =
                     glGetUniformLocation(terrain_program_id, "light_position"));
  GLint terrain_grasstex_location = 0;
  CHECK_GL_ERROR(terrain_grasstex_location =
                     glGetUniformLocation(terrain_program_id, "grasstex"));
  GLint terrain_mountaintex_location = 0;
  CHECK_GL_ERROR(terrain_mountaintex_location =
                     glGetUniformLocation(terrain_program_id, "mountaintex"));
  GLint terrain_avgh_location = 0;
  CHECK_GL_ERROR(terrain_avgh_location =
                     glGetUniformLocation(terrain_program_id, "avg_height"));


  ////////////////////////////////////////////////////////////////////////////////////

  // Switch to the Sky VAO.
  CHECK_GL_ERROR(glBindVertexArray(array_objects[kSkyVao]));

  // Generate buffer objects
  CHECK_GL_ERROR(glGenBuffers(kNumVbos, &buffer_objects[kSkyVao][0]));

  // Setup vertex data in a VBO
  CHECK_GL_ERROR(
      glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kSkyVao][kVertexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(float) * skyVerts.size() * 4,
                              &skyVerts[0], GL_STATIC_DRAW));
  CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
  CHECK_GL_ERROR(glEnableVertexAttribArray(0));

  // Setup element array buffer.
  CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                              buffer_objects[kSkyVao][kIndexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                              sizeof(uint32_t) * skyFaces.size() * 3,
                              &skyFaces[0], GL_STATIC_DRAW));

  // Triangle shaders

  // Setup vertex shader.
  GLuint sky_vertex_shader_id = 0;
  const char* sky_vertex_source_pointer = sky_vertex_shader;
  CHECK_GL_ERROR(sky_vertex_shader_id = glCreateShader(GL_VERTEX_SHADER));
  CHECK_GL_ERROR(
      glShaderSource(sky_vertex_shader_id, 1, &sky_vertex_source_pointer, nullptr));
  glCompileShader(sky_vertex_shader_id);
  CHECK_GL_SHADER_ERROR(sky_vertex_shader_id);

  // Setup geometry shader.
  GLuint sky_geometry_shader_id = 0;
  const char* sky_geometry_source_pointer = sky_geometry_shader;
  CHECK_GL_ERROR(sky_geometry_shader_id = glCreateShader(GL_GEOMETRY_SHADER));
  CHECK_GL_ERROR(
      glShaderSource(sky_geometry_shader_id, 1, &sky_geometry_source_pointer, nullptr));
  glCompileShader(sky_geometry_shader_id);
  CHECK_GL_SHADER_ERROR(sky_geometry_shader_id);

  // Setup sky fragment shader.
  GLuint sky_fragment_shader_id = 0;
  const char* sky_fragment_source_pointer = sky_fragment_shader;
  CHECK_GL_ERROR(sky_fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER));
  CHECK_GL_ERROR(glShaderSource(sky_fragment_shader_id, 1,
                                &sky_fragment_source_pointer, nullptr));
  glCompileShader(sky_fragment_shader_id);
  CHECK_GL_SHADER_ERROR(sky_fragment_shader_id);

  // Let's create our floor program.
  GLuint sky_program_id = 0;
  CHECK_GL_ERROR(sky_program_id = glCreateProgram());
  CHECK_GL_ERROR(glAttachShader(sky_program_id, sky_vertex_shader_id));
  CHECK_GL_ERROR(glAttachShader(sky_program_id, sky_fragment_shader_id));
  //CHECK_GL_ERROR(glAttachShader(sky_program_id, sky_geometry_shader_id));

  // Bind attributes.
  CHECK_GL_ERROR(glBindAttribLocation(sky_program_id, 0, "vertex_position"));
  CHECK_GL_ERROR(glBindFragDataLocation(sky_program_id, 0, "fragment_color"));
  glLinkProgram(sky_program_id);
  CHECK_GL_PROGRAM_ERROR(sky_program_id);

  // Get the uniform locations.
  GLint sky_projection_matrix_location = 0;
  CHECK_GL_ERROR(sky_projection_matrix_location =
                     glGetUniformLocation(sky_program_id, "projection"));
  GLint sky_model_matrix_location = 0;
  CHECK_GL_ERROR(sky_model_matrix_location =
                     glGetUniformLocation(sky_program_id, "model"));
  GLint sky_view_matrix_location = 0;
  CHECK_GL_ERROR(sky_view_matrix_location =
                     glGetUniformLocation(sky_program_id, "view"));
  GLint sky_light_position_location = 0;
  CHECK_GL_ERROR(sky_light_position_location =
                     glGetUniformLocation(sky_program_id, "light_position"));
  GLint sky_texture_location = 0;
  CHECK_GL_ERROR(sky_texture_location =
                     glGetUniformLocation(sky_program_id, "tex"));


  glm::vec4 light_position = glm::vec4(0.0f, 100.0f, 0.0f, 1.0f);

  std::vector<glm::vec4> subVertices;
  std::vector<glm::uvec3> subFaces;
  std::vector<glm::vec4> subNormals;
  std::vector<glm::vec2> subTerrainTypes;

  bool runThis = true;

  while (!glfwWindowShouldClose(window)) {
    // Setup some basic window stuff.
    glfwGetFramebufferSize(window, &window_width, &window_height);
    glViewport(0, 0, window_width, window_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDepthFunc(GL_LESS);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glCullFace(GL_BACK);

    if (renderWindow) {
      subVertices.clear();
      subFaces.clear();
      subNormals.clear();
      subTerrainTypes.clear();
      for(int i=0; i < mapSize*mapSize; i++)
        vertexPresentInSub[i] = -1;

      for (int i = 0; i < terrainFaces.size(); i++) {
        glm::uvec3 face = terrainFaces[i];

        glm::vec4 v1 = terrainVerts[face[0]];
        glm::vec4 v2 = terrainVerts[face[1]];
        glm::vec4 v3 = terrainVerts[face[2]];

        if (glm::length(eye - glm::vec3(v1)) < renderRadius && glm::length(eye - glm::vec3(v2)) < renderRadius && glm::length(eye - glm::vec3(v3)) < renderRadius) {
          int i1, i2, i3;
          if (vertexPresentInSub[face[0]] != -1) {
            i1 = vertexPresentInSub[face[0]];
          } else {
            subVertices.push_back(v1);
            i1 = subVertices.size() - 1;
            subNormals.push_back(vertex_normals[face[0]]);
            subTerrainTypes.push_back(vertTerrainTypes[face[0]]);
            vertexPresentInSub[face[0]] = i1;
          }
          if (vertexPresentInSub[face[1]] != -1) {
            i2 = vertexPresentInSub[face[1]];
          } else {
            subVertices.push_back(v2);
            i2 = subVertices.size() - 1;
            subNormals.push_back(vertex_normals[face[1]]);
            subTerrainTypes.push_back(vertTerrainTypes[face[1]]);
            vertexPresentInSub[face[1]] = i2;
          }
          if (vertexPresentInSub[face[2]] != -1) {
            i3 = vertexPresentInSub[face[2]];
          } else {
            subVertices.push_back(v3);
            i3 = subVertices.size() - 1;
            subNormals.push_back(vertex_normals[face[2]]);
            subTerrainTypes.push_back(vertTerrainTypes[face[2]]);
            vertexPresentInSub[face[2]] = i3;
          }

          subFaces.push_back(glm::uvec3(i1, i2, i3));
        }
      }

      // Setup vertex data in a VBO
      CHECK_GL_ERROR(
          glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kTerrainVao][kVertexBuffer]));
      CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                                  sizeof(float) * subVertices.size() * 4,
                                  &subVertices[0], GL_STATIC_DRAW));
      CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
      CHECK_GL_ERROR(glEnableVertexAttribArray(0));

      // Setup element array buffer.
      CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                                  buffer_objects[kTerrainVao][kIndexBuffer]));
      CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                                  sizeof(uint32_t) * subFaces.size() * 3,
                                  &subFaces[0], GL_STATIC_DRAW));

      // Setup vertex normal data in a VBO
      CHECK_GL_ERROR(
          glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kTerrainVao][kVertexNormalBuffer]));
      CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                                  sizeof(float) * subNormals.size() * 4,
                                  &subNormals[0], GL_STATIC_DRAW));
      CHECK_GL_ERROR(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0));
      CHECK_GL_ERROR(glEnableVertexAttribArray(1));

      // Setup vertex terrain type data in a VBO
      CHECK_GL_ERROR(
          glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kTerrainVao][kVertexTerrainTypeBuffer]));
      CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                                  sizeof(float) * subTerrainTypes.size() * 2,
                                  &subTerrainTypes[0], GL_STATIC_DRAW));
      CHECK_GL_ERROR(glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0));
      CHECK_GL_ERROR(glEnableVertexAttribArray(2));

      if (runThis) {
        CHECK_GL_ERROR(
        glBindBuffer(GL_ARRAY_BUFFER, buffer_objects[kSkyVao][kVertexBuffer]));
        CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                                    sizeof(float) * skyVerts.size() * 4,
                                    &skyVerts[0], GL_STATIC_DRAW));
        CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
        CHECK_GL_ERROR(glEnableVertexAttribArray(0));

        // Setup element array buffer.
        CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                                    buffer_objects[kSkyVao][kIndexBuffer]));
        CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                                    sizeof(uint32_t) * skyFaces.size() * 3,
                                    &skyFaces[0], GL_STATIC_DRAW));
        runThis = false;
      }

      renderWindow = false;
    }

    // Compute our view, and projection matrices.
    center = eye - camera_distance * look;

    view_matrix = glm::lookAt(eye, center, up);
    light_position = glm::vec4(eye, 1.0f);

    aspect = static_cast<float>(window_width) / window_height;
    projection_matrix =
        glm::perspective((float)(kFov * (M_PI / 180.0f)), aspect, kNear, kFar);
    model_matrix = glm::mat4(1.0f);

    moveCamera();

    glDepthMask(GL_FALSE);
    // Bind to our Sky VAO.
    CHECK_GL_ERROR(glBindVertexArray(array_objects[kSkyVao]));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureid);

    // Use our program.
    CHECK_GL_ERROR(glUseProgram(sky_program_id));

    // Pass uniforms in.
    CHECK_GL_ERROR(glUniformMatrix4fv(sky_projection_matrix_location, 1,
                                      GL_FALSE, &projection_matrix[0][0]));
    CHECK_GL_ERROR(glUniformMatrix4fv(sky_model_matrix_location, 1, GL_FALSE,
                                      &floor_model_matrix[0][0]));
    CHECK_GL_ERROR(glUniformMatrix4fv(sky_view_matrix_location, 1, GL_FALSE,
                                      &view_matrix[0][0]));
    CHECK_GL_ERROR(
        glUniform4fv(sky_light_position_location, 1, &light_position[0]));
    CHECK_GL_ERROR(glUniform1i(sky_texture_location, 0));

    // Draw our triangles.
    CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, skyFaces.size() * 3,
                                  GL_UNSIGNED_INT, 0));


    glDepthMask(GL_TRUE);
    // Bind to our Terrain VAO.
    CHECK_GL_ERROR(glBindVertexArray(array_objects[kTerrainVao]));

    // Use our program.
    CHECK_GL_ERROR(glUseProgram(terrain_program_id));

    // Pass uniforms in.
    CHECK_GL_ERROR(glUniformMatrix4fv(terrain_projection_matrix_location, 1,
                                      GL_FALSE, &projection_matrix[0][0]));
    CHECK_GL_ERROR(glUniformMatrix4fv(terrain_model_matrix_location, 1, GL_FALSE,
                                      &floor_model_matrix[0][0]));
    CHECK_GL_ERROR(glUniformMatrix4fv(terrain_view_matrix_location, 1, GL_FALSE,
                                      &view_matrix[0][0]));
    CHECK_GL_ERROR(
        glUniform4fv(terrain_light_position_location, 1, &light_position[0]));
    CHECK_GL_ERROR(glUniform1i(terrain_grasstex_location, 1));
    CHECK_GL_ERROR(glUniform1i(terrain_mountaintex_location, 2));
    CHECK_GL_ERROR(glUniform1f(terrain_avgh_location, avgFractalY));

    // Draw our triangles.
    CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, subFaces.size() * 3,
                                  GL_UNSIGNED_INT, 0));

    // Poll and swap.
    glfwPollEvents();
    glfwSwapBuffers(window);
  }
  glfwDestroyWindow(window);
  glfwTerminate();
  for (int i = 0; i < images.size(); ++i) delete[] images[i].bytes;
  exit(EXIT_SUCCESS);
}
