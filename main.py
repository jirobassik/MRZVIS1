from neural_compress_np import NeuralCompress

main_question = input("1 - Обучение\n"
                      "2 - Использование\n")

match main_question:
    case "1":
        input_img = input("Введите название картинки: ")
        block_h = int(input("Высота блока: "))
        block_w = int(input("Ширина блока: "))
        alpha = float(input("Alpha: "))
        hidden_layer = int(input("Кол-во нейронов на скрытом слое: "))
        error_m = float(input("Максимальная ошибка: "))
        au = NeuralCompress(input_img=input_img, block_h=block_h, block_w=block_w, alpha=alpha,
                            hidden_layer=hidden_layer, error_m=error_m)
        au.display_array()
        au.train()
        au.show_result()
        question = input("Сохранить веса?\n"
                         "1. Да\n"
                         "2. Нет\n")
        match question:
            case "1":
                au.save_weights()
    case "2":
        question_2 = input("1 - Сжатие расжатие\n"
                           "2 - Сжатие\n"
                           "3 - Расжатие\n")
        match question_2:
            case "1":
                input_img = input("Введите название картинки: ")
                out = NeuralCompress(input_img=input_img)
                out.display_array()
                out.show_result_use()
            case "2":
                input_img = input("Введите название картинки: ")
                block_h = int(input("Высота блока: "))
                block_w = int(input("Ширина блока: "))
                alpha = float(input("Alpha: "))
                hidden_layer = int(input("Кол-во нейронов на скрытом слое: "))
                error_m = float(input("Максимальная ошибка: "))
                au = NeuralCompress(input_img=input_img, block_h=block_h, block_w=block_w, alpha=alpha,
                                    hidden_layer=hidden_layer, error_m=error_m)
                au.train()
                file_name = input("Введите имя файла для сохранения: ")
                au.save_y(file_name)
                print("Изображение успешно зжалось")
            case "3":
                file_name = input("Введите имя файла: ")
                au = NeuralCompress()
                au.save_filename(file_name)
                au.uncompress_y()
                au.show_result_uncompress()

