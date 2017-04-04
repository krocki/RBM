/*
* @File: glWindow.h
* @Author: kamilrocki
* @Email: kmrocki@us.ibm.com
* @Created:	2015-05-12 09:11:50
* @Last Modified by:   kamilrocki
* @Last Modified time: 2017-04-04 09:16:11
*/

#ifndef __GL_WINDOW_H__
#define __GL_WINDOW_H__

#ifdef USE_OPENGL

#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <map>
#include <matrix.h>
#include <mutex>

#ifdef __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h>
#endif

#define framerate 20
#define DEFAULT_WIDTH 256
#define DEFAULT_HEIGHT 256
#define eps 10e-6

std::mutex display_mutex;
bool _pause = false;
bool _step = false;
bool learn_0 = true;
bool learn_1 = true;
bool learn_2 = true;
std::vector<float> v_err;
std::vector<float> v_hsum;
std::vector<float> v_h2_sum;

typedef struct {

	GLint x;
	GLint y;
	GLint w;
	GLint h;

} window_properties;

std::map<std::string, window_properties> w_props;
std::map<int, Matrix<float>*> windows;

void draw_filled_rectangle(float x, float y, float h, float w, float r, float g, float b, float a) {

	glColor4f(r, g, b, a);

	glBegin(GL_QUADS);

	//margins, top frame
	glVertex2i((int)(x), (int)(y));
	glVertex2i((int)(x + w), (int)(y));
	glVertex2i((int)(x + w), (int)(y + h));
	glVertex2i((int)(x), (int)(y + h));

	glEnd();

}

void draw_grid(float r, float g, float b, float a, float cells_h, float cells_v) {

	GLint h = glutGet(GLUT_WINDOW_HEIGHT);
	GLint w = glutGet(GLUT_WINDOW_WIDTH);

	float scale_x = (float)w / (float)(cells_h);
	float scale_y = (float)h / (float)(cells_v);

	glColor4f(r, g, b, a);
	glBegin(GL_LINES);

	for (unsigned i = 1; i < cells_h; i++) {

		glVertex2i((int)((float)i * scale_x), 0);
		glVertex2i((int)((float)i * scale_x), (GLint)h);

	}

	for (unsigned i = 1; i < cells_v; i++) {

		glVertex2i(0, (int)((float)i * scale_y));
		glVertex2i((GLint)w, (int)((float)i * scale_y));

	}

	glEnd();
}

void draw_text(float x, float y, char* string, float r, float g, float b, float a, void* font) {

	int len, i;

	glColor4f(r, g, b, a);

	glRasterPos2i(static_cast<int>(x), static_cast<int>(y));

	len = static_cast<int>(strlen(string));

	for (i = 0; i < len; i++)
		glutBitmapCharacter(font, string[i]);

}

void plot_graph(std::vector<float> data, char* legend, float thickness, float x_spacing, float r, float g, float b, float a, GLushort pattern, bool autoscale) {

	GLint h = glutGet(GLUT_WINDOW_HEIGHT);
	GLint w = glutGet(GLUT_WINDOW_WIDTH);
	GLint factor = 1;    // Stippling factor

	float x_margin = 0.8;
	float y_margin = 0.8;
	float range_increment = 0.1f;
	double range_min = 0.0f;
	double range_max = 1.0f;

	if (autoscale && data.size() > 0) {

		size_t elements_visible = w * x_margin / x_spacing;
		elements_visible = __min(data.size(), elements_visible);
		double maxval =  *std::max_element(data.end() - elements_visible, data.end());
		double minval =  *std::min_element(data.end() - elements_visible, data.end());
		range_min = minval - eps;
		range_max = maxval + eps;
		range_increment = range_max - range_min;
	}

	int acc_x = (1.0f - x_margin) / 2.0f * w;
	int acc_y = (1.0f - y_margin) / 2.0f * h;

	glLineWidth(thickness);

	for (double i = range_min; i <= range_max + 1; i += range_increment) {

		double bh = (i - range_min) / (range_max - range_min);

		char value[10];
		//sprintf(value, "%.5f", (float)(i));
		draw_text(acc_x - 30, acc_y + h * y_margin - (int)(bh * h * y_margin) + 2, value, r, g, b, a, GLUT_BITMAP_HELVETICA_10);
		draw_text(acc_x + w * x_margin + 5, acc_y + h * y_margin - (int)(bh * h * y_margin) + 2, value, r, g, b, a, GLUT_BITMAP_HELVETICA_10);
		glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
		glBegin(GL_LINES);
		glVertex2i(acc_x, acc_y + h * y_margin - (int)(bh * h * y_margin));
		glVertex2i(acc_x + w * x_margin, acc_y + h * y_margin - (int)(bh * h * y_margin));
		glEnd();

	}

	glEnable(GL_LINE_STIPPLE);

	glLineStipple(factor, pattern);
	glColor4f(r, g, b, a);

	glBegin(GL_LINE_STRIP);

	for (int i = 0; i < w * x_margin / x_spacing; i++) {

		if (data.size() - 1 - (unsigned)(int)i < data.size()) {

			float d = data[data.size() - 1 - (unsigned)(int)(i)];
			float dh = (d - range_min) / (range_max - range_min);
			glVertex2i(acc_x + i * x_spacing, acc_y + h * y_margin - (int)((dh) * h * y_margin));
		}

	}

	glEnd();

	glDisable(GL_LINE_STIPPLE);

	int i = 0;

	if (data.size() - 1 - (unsigned)(int)i < data.size()) {

		float d = data[data.size() - 1 - (unsigned)(int)(i)];
		float dh = (d - range_min) / (range_max - range_min);
		char value[20];
		sprintf(value, "%s %.5f", legend, d);
		draw_text(acc_x + i * x_spacing - 80, acc_y + h * y_margin - (int)((dh) * h * y_margin), value, r, g, b, a, GLUT_BITMAP_HELVETICA_10);

	}

	i = __min(data.size() - 1, w * x_margin / x_spacing);

	if (data.size() - 1 - (unsigned)(int)i < data.size()) {

		float d = data[data.size() - 1 - (unsigned)(int)(i)];
		float dh = (d - range_min) / (range_max - range_min);
		char value[10];
		sprintf(value, "%.5f", d);
		draw_text(acc_x + i * x_spacing + 10, acc_y + h * y_margin - (int)((dh) * h * y_margin), value, r, g, b, a, GLUT_BITMAP_HELVETICA_10);

	}

}

class GLWindowRenderer {

  public:

	static void init(void) {

		char *argv[1];
		int argc = 1;
		argv [0] = "GLWindowRenderer";

		glutInit(&argc, argv);
		load_properties_from_file();

	}

	static void load_properties_from_file(void) {

		w_props.insert({"v", {30, 30, 200, 200}});
		w_props.insert({"vn", {280, 30, 200, 200}});
		w_props.insert({"pc", {530, 30, 200, 200}});
		w_props.insert({"H", {780, 30, 200, 200}});
		w_props.insert({"ve", {980, 30, 420, 200}});
		//w_props.insert({"h_act", {680, 120, 120, 120}});
		w_props.insert({"W", {30, 300, 1000, 1000}});
		// FILE * file = fopen("prefs.dat", "rb");

		// if (file != NULL) {

		//     fread(&w_props, sizeof(w_props), 1, file);
		//     fclose(file);

		// }

	}

	static void save_properties_to_file(void) {

		// FILE * file = fopen("prefs.dat", "wb");

		// if (file != NULL) {

		//     fwrite(&w_props, sizeof(w_props), 1, file);
		//     fclose(file);

		// }

	}

	static void keyboard(unsigned char key, int x, int y) {

		switch (key) {

		case ' ': // space
			_pause = !_pause;
			break;
		case '>': // next
		case '.':
			_step = true;
			break;
		case '1': // 1
			learn_1 = !learn_1;
			printf("1: %d\n", learn_1);
			break;
		case '2': // 2
			learn_2 = !learn_2;
			printf("2: %d\n", learn_2);
			break;
		case '0': // 2
			learn_0 = !learn_0;
			printf("0: %d\n", learn_0);
			break;
		case 27: // Escape key
			printf("OpenGL: ESC signal...\n");

			// don't allow other threads to do anything
			display_mutex.lock();
			for (std::map<int, Matrix<float>*>::iterator it = windows.begin(); it != windows.end(); ++it) {
				printf("-%d\n", it->first);
				glutDestroyWindow(it->first);
			}

			save_properties_to_file();
			windows.clear();
			exit(0);
			//

		}
	}

	static void display(void) {

		display_mutex.lock();

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//display_image(pi, w, h, preview_show_grid, false);
		Matrix<float> * im = windows.find(glutGetWindow())->second;
		//printf("display_%d\n", glutGetWindow());
		if (im == nullptr)
			std::cout << "null" << std::endl;

		//display matrix

		if (im != nullptr) {

			if (im->display_mode == DISPLAY_MODE::GRAPH) {

				plot_graph(v_err, "h0 error", 3.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0xFFFF, true);

			} else if (im->display_mode == DISPLAY_MODE::ROWS_ARE_IMAGES ||
			           im->display_mode == DISPLAY_MODE::ROWS_ARE_WEIGHTS) {

				float h = sqrtf(im->dim[0] * im->dim[1]);
				float w = h;
				unsigned im_w = sqrtf(im->dim[1]);
				unsigned images = sqrtf(im->dim[0]);

				//printf("display: dim %zu x %zu\n", im->dim[0], im->dim[1]);

				float scale_x = (float)glutGet(GLUT_WINDOW_HEIGHT) / (float)(h);
				float scale_y = (float)glutGet(GLUT_WINDOW_WIDTH) / (float)(w);

				for (unsigned y = 0; y < im->dim[0]; y++) {

					float offset_x = y % images;
					float offset_y = y / images;

					offset_x *= im_w;
					offset_y *= im_w;

					float l2 = 0.0f;

					for (unsigned x = 0; x < im->dim[1]; x++) {

						float val = (*im)(y, x);
						l2 += val * val;
					}

					l2 = sqrtf(l2);

					for (unsigned x = 0; x < im->dim[1]; x++) {

						float val = (*im)(y, x);

						if (im->display_mode == DISPLAY_MODE::ROWS_ARE_WEIGHTS) {

							//normalize
							val /= l2;
							val += 0.5f;


						}

						float new_x = offset_x + x % im_w;
						float new_y = offset_y + x / im_w;

						draw_filled_rectangle(float(new_x) * scale_y, float(new_y) * scale_x, scale_x, scale_y,
						                      (float)val,
						                      (float)val,
						                      (float)val,
						                      (float)1.0f);

					}

				}

				draw_grid(0.5f, 0.3f, 0.3f, 0.3f, images, images);

			} else {

				size_t temp_dim[3];

				if (im->display_mode == DISPLAY_MODE::UNFLATTEN) {

					temp_dim[0] = im->dim[0];
					temp_dim[1] = im->dim[1];
					temp_dim[2] = im->dim[2];

					im->dim[0] = (GLint)sqrtf(im->elements * temp_dim[0]);
					im->dim[1] = (GLint)sqrtf(im->elements / temp_dim[0]);
				}

				float scale_x = (float)glutGet(GLUT_WINDOW_HEIGHT) / (float)(im->dim[0]);
				float scale_y = (float)glutGet(GLUT_WINDOW_WIDTH) / (float)(im->dim[1]);

				//printf("display: dim %zu x %zu\n", im->dim[0], im->dim[1]);

				for (unsigned y = 0; y < im->dim[0]; y++) {

					for (unsigned x = 0; x < im->dim[1]; x++) {

						float val = (*im)(y, x);

						draw_filled_rectangle(float(x) * scale_y, float(y) * scale_x, scale_x, scale_y,
						                      (float)val,
						                      (float)val,
						                      (float)val,
						                      (float)1.0f);

					}

				}

				draw_grid(0.3f, 0.3f, 0.3f, 0.1f, im->dim[1], im->dim[0]);

				if (im->display_mode == DISPLAY_MODE::UNFLATTEN) {

					im->dim[0] = temp_dim[0];
					im->dim[1] = temp_dim[1];
					im->dim[2] = temp_dim[2];
				}

			}
		}

		display_mutex.unlock();

		glutSwapBuffers();
	}

	static void add(std::string name) {

		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

		GLint x = (glutGet(GLUT_SCREEN_WIDTH) - DEFAULT_WIDTH) / 2;
		GLint y = (glutGet(GLUT_SCREEN_HEIGHT) - DEFAULT_HEIGHT) / 2;
		GLint w = DEFAULT_WIDTH;
		GLint h = DEFAULT_HEIGHT;

		std::map<std::string, window_properties>::iterator it = w_props.find(name);

		if (it != w_props.end()) {

			window_properties props = it->second;
			x = props.x;
			y = props.y;
			w = props.w;
			h = props.h;

		}

		glutInitWindowPosition(x, y);
		glutInitWindowSize(w, h);

		GLint id = glutCreateWindow(name.c_str());
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutReshapeFunc(reshape);
		// glutReshapeFunc(reshape);
		// glutMouseFunc(mouse_func);
		// glutMotionFunc(motion_func);
		// glutPassiveMotionFunc(pmotion_func);
		windows.insert ( {id, nullptr});
		w_props.insert({name, {x, y, w, h}});
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	}

	// static Matrix<float>*p get(int id) {

	// 	return windows.find(id)->second;
	// }

	static void set(int id, Matrix<float>*p) {

		windows.find(id)->second = p;
	}

	static void reshape(int width, int height) {

		glViewport(0, 0, width, height);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		//set the coordinate system, with the origin in the top left
		gluOrtho2D(0, width, height, 0);
		glMatrixMode(GL_MODELVIEW);

	}

	static void idle(void) {

		//for all windows in the list

		usleep(1000000.0 / framerate);
		for (std::map<int, Matrix<float>*>::iterator it = windows.begin(); it != windows.end(); it++) {

			//printf("glutPostRedisplay_%d\n", it->first);
			if (it->second != nullptr) {

				glutSetWindow(it->first);
				glutPostRedisplay();

			}

		}

	}

	static void main_loop(void) {

		glutIdleFunc(idle);
		glutMainLoop();

	}

};

#endif

#endif
