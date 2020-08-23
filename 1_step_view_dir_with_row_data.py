"""
    1)Распечатывает в stdout структуру папки, расширения файлов
    2)ко всем функция добавляется название __make_dataset__ в качетсве сишной реализации namespace
    3)Список использованных функций:
        1)__make_dataset__scan_folder()
        2)__make_dataset__nice_ls()
        3)__make_dataset__print_extensions_of_the_files()
        4)__make_dataset__execution_of_existing_functions()
    4)Список неипользованных функций в конрентной задаче MNIST
        1)__make_dataset__get_file_paths()

"""

import os
import json


def __make_dataset__scan_folder(path, max_entries=4):
    """
        Возращает список.

        1) если по указанному пути лежат просто файлы, то функция вернет список имен файлов
        2) если по указанному пути лежат папки, то функция вернет список словарей
        3) для простой и понятной визуализации в терминале рекомендуется использовать json
        4) внутри списка могут находиться как названия файлов, так и словари, описывающие вложенную структуру папок
        5) глубина рекурсии не обрабатывается
    """

    entries = os.listdir(path)

    if not entries:
        # Пустой каталог - возвращаем None
        return None

    if len(entries) > max_entries:
        result = []
        i = 0
        for entry in entries:
            if i == max_entries:
                break

            entry_path = path + '/' + entry
            if os.path.isfile(entry_path):
                # Для файла - добавляем в лист его имя в виде строки
                result.append(entry)
            else:
                # Для каталога - добавляем в лист его имя в виде Dict
                result.append(
                    {entry: __make_dataset__scan_folder(entry_path, max_entries)}
                )
            i += 1
        return result

    if 1 < len(entries) <= max_entries:
        # Больше 2 записей в каталоге - возвращаем List
        result = []
        for entry in entries:
            entry_path = path + '/' + entry
            if os.path.isfile(entry_path):
                # Для файла - добавляем в лист его имя в виде строки
                result.append(entry)
            else:
                # Для каталога - добавляем в лист его имя в виде Dict
                result.append(
                    {entry: __make_dataset__scan_folder(entry_path, max_entries)}
                )
        return result

    # При одной записи в каталоге:
    entry = entries[0]
    entry_path = path + '/' + entry
    if os.path.isfile(entry_path):
        # Для файла возвращаем его имя в виде строки
        return entry
    else:
        # Для каталога - возвращаем имя в виде Dict
        return {entry: __make_dataset__scan_folder(entry_path, max_entries)}


def __make_dataset__nice_ls(path, max_entries=4):
    """
        обертка для простого print в stdout в формате json результата работы функции scan_folder
    """

    dir_map = {path: __make_dataset__scan_folder(path, max_entries)}
    print("########################")
    print("structure_of_the_folder:\n")
    print(json.dumps(dir_map, indent=10, sort_keys=False).replace(": null", ": None"))
    print("########################")


def __make_dataset__print_extensions_of_the_files(path):
    """
        в stdout печает список всех типов файлов, лежащих в папке path
        1)список хранит в себе только уникальные значения
        2)без ограничения не длину списка
        3)если в названии файла нету его расширения например ".txt", то функция рапечатает None
    """

    list_of_the_files_extensions = []
    for root, dirs, files in os.walk(path):
        for file in files:
            index_of_symbol = file.find('.')
            if index_of_symbol == -1:
                list_of_the_files_extensions.append('None')

            file = file[index_of_symbol:]
            list_of_the_files_extensions.append(file)

    list_with_unique_values = []

    for filename in list_of_the_files_extensions:
        if len(list_with_unique_values) == 0:
            list_with_unique_values.append(filename)
        else:
            for i in range(len(list_with_unique_values)):
                if filename == list_with_unique_values[i]:
                    continue
                else:
                    list_with_unique_values.append(filename)

    extension_map = {path: list_with_unique_values}
    print("########################")
    print("extensions_of_the_files:\n")
    print(json.dumps(extension_map, indent=10, sort_keys=False).replace(": null", ": None"))
    print("########################")


def make_dataset_execution_of_existing_functions(path, max_entries=4):
    """
        выполнение nice_ls и print_extensions_of_the_files
    """
    __make_dataset__nice_ls(path, max_entries)
    __make_dataset__print_extensions_of_the_files(path)


def make_dataset_get_file_paths(path):
    """список абсолютных путей к файлам в директории path"""
    file_paths = []

    for root, dirs, files in os.walk(path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    return file_paths


if __name__ == '__main__':
    make_dataset_execution_of_existing_functions('F:\\datasets\\mnist', 10)
