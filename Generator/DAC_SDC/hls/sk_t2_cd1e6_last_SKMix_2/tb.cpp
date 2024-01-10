/********************************************************************************
* Filename: tb.cpp
* Date: Tue Jan  9 15:33:29 2024
* Description: reference testbench for accelerator
********************************************************************************/
#include <stdint.h>
#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <string>
#include "stream_tools.h"
#include "config.h"
using namespace std;

#define grid_row 20
#define grid_col 40
#define org_row 360
#define org_col 640
#define inp_row 160
#define inp_col 320
#define div 1829.2285204780505


string opath = "E:/Projects/DeepBurning_MixQ/DAC_SDC_tests/2_skynet_mixed2/debug_path/";

void sky_net(stream<my_ap_axis >& in, stream<my_ap_axis >& out, const unsigned int reps);

void load_data(const char *path, char *ptr, unsigned int size) {
  std::ifstream f(path, std::ios::in | std::ios::binary);
  if (!f) {
    std::cout << "no such file,please check the file name!/n";
    exit(0);
  }
  f.read(ptr, size);
  f.close();
}

void write_data(const char *path, char *ptr, unsigned int size) {
  std::ofstream f(path, std::ios::out | std::ios::binary);
  if (!f) {
    std::cout << "write no such file,please check the file name!/n";
    exit(0);
  }
  f.write(ptr, size);
  f.close();
}



int main(int argc, char const *argv[])
{
    unsigned char img[inp_row][inp_col][3];

    load_data("E:/Projects/DeepBurning_MixQ/DAC_SDC_tests/test_data/0.bin", (char *) img, sizeof(img));

    unsigned char * data = (unsigned char *) img;
    const int data_points_per_line = 8;        // ch * 10
    const int nums_line_pre_img = inp_row * inp_col * 3 / 8;

    int img_repeat = 1;

    hls::stream<my_ap_axis> input_stream("input stream");
    for (unsigned int rp = 0; rp < img_repeat; rp++)
    {
    	for (unsigned int i = 0; i < nums_line_pre_img; i++) {
    		my_ap_axis temp;
    		for (unsigned int j = 0; j < data_points_per_line; j++) {
    			temp.data( 8*(j+1)-1, 8*j ) = data[i * data_points_per_line + j];
    		}
    		input_stream.write(temp);
    	}
    }

    hls::stream<my_ap_axis> output_stream("output stream");
    sky_net(input_stream, output_stream, img_repeat);

    cout << "output size :" << output_stream.size() << endl;
    
    for (unsigned ir = 0; ir < img_repeat; ir++){
        ap_int<32> conv_last_out [grid_row*grid_col][6][6];

        for(unsigned r = 0; r < grid_row; r++)
            for(unsigned ofi = 0; ofi < 18; ofi++)
                for(unsigned c = 0; c < grid_col; c++){
                    my_ap_axis output = output_stream.read();
                    ap_uint<64> out_data = output.data;
                    conv_last_out[c + r*grid_col][ofi / 3][(ofi % 3)*2]     = out_data(31,0);
                    conv_last_out[c + r*grid_col][ofi / 3][(ofi % 3)*2 + 1] = out_data(63,32);
                }

        int conf [grid_row*grid_col] = {0};
        for(unsigned int i = 0; i< (grid_row*grid_col); i++)
            for(unsigned int j = 0; j<6; j++){
                conf[i] += conv_last_out[i][j][4];
            }
  
        unsigned int max_index = 0;

        int max = -9999999;
        for(unsigned int i = 0; i< (grid_row*grid_col); i++)
        {
            if(conf[i] > max){
                max = conf[i];
                max_index = i;
            }
        }
        

        unsigned int grid_x;
        unsigned int grid_y;
        grid_x = max_index % grid_col;
        grid_y = max_index / grid_col;

        float boxs[6][4];
        for(unsigned int i = 0; i<6; i++)
            for(unsigned int j = 0; j<4; j++){
                boxs[i][j] = float(conv_last_out[max_index][i][j]) / div;
            }

        float x = 0, y = 0, w = 0, h = 0;
        for(unsigned int i = 0; i<6; i++){
            x += 1 / (1 + std::exp(-boxs[i][0]));
            y += 1 / (1 + std::exp(-boxs[i][1]));
            w += std::exp(boxs[i][2]);
            h += std::exp(boxs[i][3]);
        }
        x = x / 6;
        y = y / 6;
        w = w / 6;
        h = h / 6;

        x = (x + grid_x) * 8;
        y = (y + grid_y) * 8;
        w = w*20;
        h = h*20;

        float xmin,xmax,ymin,ymax;

        xmin = (x - w/2)*org_col/inp_col;
        xmax = (x + w/2)*org_col/inp_col;
        ymin = (y - h/2)*org_row/inp_row;
        ymax = (y + h/2)*org_row/inp_row;

        cout << "x: " << x << " y: " << y << " w: " << w << " h: " << h << endl;
        cout << "xmin: " << xmin << " xmax: " << xmax << " ymin: " << ymin << " ymax: " << ymax << endl;
    }

    return 0;
}
