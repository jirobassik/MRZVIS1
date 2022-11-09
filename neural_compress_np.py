import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from json_func import json_load, json_save


class NeuralCompress:
    def __init__(self, input_img="pictures//pic_1.png", block_h=4, block_w=4, alpha=0.001,
                 hidden_layer=20, error_m=2500.0):
        self.img = mpimg.imread(input_img)

        self.img_height = 256
        self.img_width = 256

        self.img_array = numpy.array(self.img_to_array(self.img))

        self.block_height = block_h
        self.block_width = block_w
        self.alpha = alpha

        self.number_of_blocks = int((self.img_height * self.img_width) / (self.block_height * self.block_width))

        self.input_layer_size = self.block_height * self.block_height * 3
        self.hidden_layer_size = hidden_layer

        self.blocks = self.__to_blocks().reshape((self.number_of_blocks, 1, self.input_layer_size))

        self.first_layer = numpy.random.rand(self.input_layer_size, self.hidden_layer_size) * 2 - 1

        temp_layer = numpy.copy(self.first_layer)
        self.second_layer = temp_layer.transpose()

        self.file_name = None

        self.result = None
        self.y_list = []
        self.error_max = error_m

    def __get_block(self, num_img, num_width):
        block = []
        for num_block_height in range(self.block_height):
            for num_block_weight in range(self.block_width):
                for color in range(3):
                    block.append(self.img_array[num_img * self.block_height + num_block_height,
                                                num_width * self.block_width + num_block_weight, color])
        return block

    def __to_blocks(self):
        blocks = []
        for num_img in range(self.img_height // self.block_height):
            for num_width in range(self.img_width // self.block_width):
                block = self.__get_block(num_img, num_width)
                blocks.append(block)
        return numpy.array(blocks)

    def __to_array(self, blocks):
        array = []
        blocks_in_line = self.img_width // self.block_width
        for num_img_h in range(self.img_height // self.block_height):
            for num_bl_h in range(self.block_height):
                line = []
                for num_bl_in_l in range(blocks_in_line):
                    self.__sub_array(num_bl_in_l, blocks_in_line, num_img_h, num_bl_h, blocks, line)
                array.append(line)
        return numpy.array(array)

    def __sub_array(self, num_bl_in_l, blocks_in_line, num_img_h, num_bl_h, blocks, line):
        for num_bl_w in range(self.block_width):
            pixel = []
            for color in range(3):
                pixel.append(
                    blocks[num_img_h * blocks_in_line + num_bl_in_l,
                           (num_bl_h * self.block_width * 3) + (num_bl_w * 3) + color])
            line.append(pixel)

    @staticmethod
    def __display_array(img_array):
        img_array = 1 * (img_array + 1) / 2
        figsize = 256 / float(100), 256 / float(100)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img_array)
        ax.axis("off")
        plt.savefig("save_figure.png")
        plt.show()

    @staticmethod
    def img_to_array(img_):
        return (2.0 * img_ / 1.0) - 1.0

    @staticmethod
    def normalize_matrix(matrix):
        for num_row in range(len(matrix[0])):
            s = 0
            for elem in range(len(matrix)):
                s += matrix[elem][num_row] * matrix[elem][num_row]
            s = math.sqrt(s)
            for elem in range(len(matrix)):
                matrix[elem][num_row] = matrix[elem][num_row] / s

    @staticmethod
    def load_weights_():
        loaded_arr_w1, loaded_arr_w2 = numpy.loadtxt("save_weights_1.txt"), numpy.loadtxt("save_weights_2.txt")
        return loaded_arr_w1, loaded_arr_w2

    @staticmethod
    def load_weights_w1_w2():
        loaded_arr_w1, loaded_arr_w2 = numpy.loadtxt("save_weights_w1.txt"), numpy.loadtxt("save_weights_w2.txt")
        return loaded_arr_w1, loaded_arr_w2

    def read_y(self):
        loaded_arr = numpy.loadtxt(self.file_name)
        load_compr_image = loaded_arr.reshape((
            loaded_arr.shape[0], loaded_arr.shape[1] // json_load()["yshape"], json_load()["yshape"]))
        return load_compr_image

    def save_filename(self, file):
        self.file_name = file

    def save_weights(self):
        arr_reshaped_w1 = self.first_layer.reshape(self.first_layer.shape[0], -1)
        arr_reshaped_w2 = self.second_layer.reshape(self.second_layer.shape[0], -1)
        numpy.savetxt("save_weights_1.txt", arr_reshaped_w1)
        numpy.savetxt("save_weights_2.txt", arr_reshaped_w2)

    def save_weights_for_compr(self):
        arr_reshaped_w1 = self.first_layer.reshape(self.first_layer.shape[0], -1)
        arr_reshaped_w2 = self.second_layer.reshape(self.second_layer.shape[0], -1)
        numpy.savetxt("save_weights_w1.txt", arr_reshaped_w1)
        numpy.savetxt("save_weights_w2.txt", arr_reshaped_w2)

    def display_array(self):
        self.img_array = 1 * (self.img_array + 1) / 2
        figsize = self.img_height / float(100), self.img_width / float(100)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(self.img_array)
        ax.axis("off")
        plt.show()

    def save_y(self, file_name):
        self.y_list = numpy.array(self.y_list)
        arr_reshaped = self.y_list.reshape(self.y_list.shape[0], -1)
        numpy.savetxt(file_name, arr_reshaped)
        json_save(self.y_list.shape[2], "yshape")

    def uncompress_y(self):
        first_layer, second_layer = self.load_weights_w1_w2()
        compr_image = self.read_y()
        decompress = []
        for block in compr_image:
            decompress.append(
                numpy.matmul(
                    numpy.matmul(block, first_layer),
                    second_layer))
        decompress = numpy.array(decompress)
        return decompress

    def restore_uncompress(self):
        return self.__to_array(self.uncompress_y().reshape(
            self.number_of_blocks, self.input_layer_size))

    def show_result_uncompress(self):
        self.__display_array(self.restore_uncompress())

    def train(self):
        error_max = self.error_max
        error_current = error_max + 1
        epoch = 0
        while error_current > error_max:
            error_current = 0
            epoch += 1
            for vect_bl in self.blocks:
                # Y(i) = X(i)*W
                y = vect_bl @ self.first_layer
                # X` (i) = Y(i)*W`
                x1 = y @ self.second_layer
                # ∆X(i) = X`(i) – X(i)
                dx = x1 - vect_bl
                # W`(t + 1) = W`(t) – a`*[Y(i)] T *∆X(i),
                self.first_layer -= self.alpha * numpy.matmul(
                    numpy.matmul(vect_bl.transpose(), dx),
                    self.second_layer.transpose())
                self.normalize_matrix(self.first_layer)
                self.second_layer -= self.alpha * numpy.matmul(y.transpose(), dx)
                self.normalize_matrix(self.second_layer)

            error_current = self.calc_error(error_current)
            print('Epoch:', epoch)
            print('Error:', error_current)

        print('Z:', self.calc_compression_rate())

        for vect_bl in self.blocks:
            self.y_list.append(vect_bl)
        self.save_weights_for_compr()

    def calc_error(self, error_cur):
        for vect_bl in self.blocks:
            dx = ((vect_bl @ self.first_layer) @ self.second_layer) - vect_bl
            # Е(q) = ∑ ∆X(q)i *∆X(q)i , где 1 <= i <= N
            error = (dx * dx).sum()
            error_cur += error
        return error_cur

    def calc_compression_rate(self):  # Коэффициент сжатия, Z = (N*L)/((N+L)*p+2)
        return ((numpy.size(self.blocks) * self.number_of_blocks) / (
                (numpy.size(self.blocks) + self.number_of_blocks) * self.hidden_layer_size + 2)) / 100

    def restore_img_use(self):
        result = []
        first_layer, second_layer = self.load_weights_()
        for block in self.blocks:
            result.append(block.dot(first_layer).dot(second_layer))
        result = numpy.array(result)
        return self.__to_array(result.reshape(
            self.number_of_blocks, self.input_layer_size))

    def show_result_use(self):
        self.__display_array(self.restore_img_use())

    def restore_img(self):
        self.result = []
        for block in self.blocks:
            self.result.append(block.dot(self.first_layer).dot(self.second_layer))
        self.result = numpy.array(self.result)
        return self.__to_array(self.result.reshape(
            self.number_of_blocks, self.input_layer_size))

    def show_result(self):
        self.__display_array(self.restore_img())
