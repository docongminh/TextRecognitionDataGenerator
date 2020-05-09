import argparse
import os, errno
import random
import re
import requests
import string

from bs4 import BeautifulSoup
from PIL import Image, ImageFont, ImageFile
from data_generator import FakeTextDataGenerator
from multiprocessing import Pool
ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob
from itertools import chain
import sys
import math
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
import cv2

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Generate synthetic text data for text recognition.')
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default="images/",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        default=""
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), or cn (Chinese).",
        default="vie"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="The number of images to be created.",
        default=1000
    )
    parser.add_argument(
        "-rs",
        "--random_sequences",
        action="store_true",
        help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
        default=False
    )
    parser.add_argument(
        "-rsff",
        "--random_sequences_from_font",
        action="store_true",
        help="Use random sequences by characters in random fonts as the source text for the generation.",
        default=False
    )
    parser.add_argument(
        "-sjnk",
        "--random_sequences_sjnk",
        action="store_true",
        help="Generate data for sjnk project",
        default=False
    )
    parser.add_argument(
        "-sjnk_latin",
        "--random_latin_sjnk",
        action="store_true",
        help="Generate data for sjnk project",
        default=False
    )
    parser.add_argument(
        "-let",
        "--include_letters",
        action="store_true",
        help="Define if random sequences should contain letters. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-num",
        "--include_numbers",
        action="store_true",
        help="Define if random sequences should contain numbers. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-sym",
        "--include_symbols",
        action="store_true",
        help="Define if random sequences should contain symbols. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
        default=1
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        default=False
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        help="Define the height of the produced images",
        default=32,
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        help="Define the number of thread to use for image generation",
        default=16,
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        default="jpg",
    )
    parser.add_argument(
        "-pre",
        "--prefix",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        default="",
    )
    parser.add_argument(
        "-k",
        "--skew_angle",
        type=float,
        nargs="?",
        help="Define skewing angle of the generated text. In positive degrees",
        default=0,
    )
    parser.add_argument(
        "-rk",
        "--random_skew",
        action="store_true",
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
        default=False,
    )
    parser.add_argument(
        "-wk",
        "--use_wikipedia",
        action="store_true",
        help="Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s",
        default=False,
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=int,
        nargs="?",
        help="Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius",
        default=0,
    )
    parser.add_argument(
        "-rbl",
        "--random_blur",
        action="store_true",
        help="When set, the blur radius will be randomized between 0 and -bl.",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--background",
        type=int,
        nargs="?",
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures",
        default=0,
    )
    parser.add_argument(
        "-hw",
        "--handwritten",
        action="store_true",
        help="Define if the data will be \"handwritten\" by an RNN",
    )
    parser.add_argument(
        "-na",
        "--name_format",
        type=int,
        help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings",
        default=0,
    )
    parser.add_argument(
        "-d",
        "--distorsion",
        type=int,
        nargs="?",
        help="Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
        default=0
    )
    parser.add_argument(
        "-do",
        "--distorsion_orientation",
        type=int,
        nargs="?",
        help="Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
        default=0
    )

    return parser.parse_args()

def load_dict(lang):
    """
        Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(os.path.join('dicts', lang + '.txt'), 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = d.readlines()
    return lang_dict

def load_fonts(lang):
    """
        Load all fonts in the fonts directories
    """
    # if lang == 'cn':
    return glob.glob('fonts/{0}/*.ttf'.format(lang)) + \
           glob.glob('fonts/{0}/*.ttc'.format(lang)) + \
           glob.glob('fonts/{0}/*.otf'.format(lang))
    # else:
    #     return [os.path.join('fonts/latin', font) for font in os.listdir('fonts/latin')]

def create_strings_from_file(filename, count):
    """
        Create all strings by reading lines in specified files
    """

    strings = []

    with open(filename, 'r', encoding="utf8") as f:
        lines = [l.strip()[0:200] for l in f.readlines()]
        if len(lines) == 0:
            raise Exception("No lines could be read in file")
        while len(strings) < count:
            if len(lines) > count - len(strings):
                strings.extend(lines[0:count - len(strings)])
            else:
                strings.extend(lines)

    return strings

def create_strings_from_dict(length, allow_variable, count, lang_dict):
    """
        Create all strings by picking X random word in the dictionnary
    """

    dict_len = len(lang_dict)
    strings = []
    for _ in range(0, count):
        current_string = ""
        for _ in range(0, random.randint(1, length) if allow_variable else length):
            current_string += lang_dict[random.randrange(dict_len)][:-1]
            current_string += ' '
        strings.append(current_string[:-1])
    return strings

def create_strings_from_wikipedia(minimum_length, count, lang):
    """
        Create all string by randomly picking Wikipedia articles and taking sentences from them.
    """
    sentences = []

    while len(sentences) < count:
        # We fetch a random page
        page = requests.get('https://{}.wikipedia.org/wiki/Special:Random'.format(lang))

        soup = BeautifulSoup(page.text, 'html.parser')

        for script in soup(["script", "style"]):
            script.extract()

        # Only take a certain length
        lines = list(filter(
            lambda s:
                len(s.split(' ')) > minimum_length
                and not "Wikipedia" in s
                and not "wikipedia" in s,
            [
                ' '.join(re.findall(r"[\w']+", s.strip()))[0:80] for s in soup.get_text().splitlines()
            ]
        ))

        # Remove the last lines that talks about contributing
        sentences.extend(lines[0:max([1, len(lines) - 5])])

    return sentences[0:count]
def create_strings_from_fonts(fonts):
    strings = []
    font_dicts = {}
    for font in fonts:
        if (font not in font_dicts):
            ttf = TTFont(font, fontNumber=0)

            chars = [u'{0}'.format(chr(x[0])) for x in
                     list(chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables))]
            font_dicts[font] = chars
        else:
            chars = font_dicts[font]

        strings.append(''.join(random.choice(chars) for i in range(random.randint(0, 100))))
    return strings

def check_character_in_font(char, font):
    try:
        for cmap in font['cmap'].tables:
            if cmap.isUnicode():
                if ord(char) in cmap.cmap:
                    return True
    except Exception as ex:
        print(ex)
        print(u'1{}1'.format(char))
    return False

def check_character_in_fontc1(char, font, height = 32):
    image_font = ImageFont.truetype(font=font, size=height)
    text_width, text_height = image_font.getsize(char)

    # text_height = random.randint(text_height, text_height + 30)
    # text = u'日産コーポレート/個人ゴールドJBC123JAL'
    txt_img = Image.new('L', (text_width, text_height), 255)

    txt_draw = ImageDraw.Draw(txt_img)

    txt_draw.text((0, 0), u'{0}'.format(char), fill=0, font=image_font)

    txt_img = np.array(txt_img)
    # cv2.imshow("img", txt_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # txt_img = cv2.cvtColor(txt_img, cv2.COLOR_RGB2GRAY)
    return len(np.nonzero(np.array(txt_img) != 255)[0]) > 0

def random_latin(fonts):
    strings = []
    latin_chars = [x[:-1] for x in open("dicts/latin.txt", encoding="utf-8").readlines()]
    special_chars = [x[:-1] for x in open("dicts/special_char.txt", encoding="utf-8").readlines()][:-3]
    max_length = 60

    all_chars = latin_chars + special_chars

    for font in fonts:
        generated = ""
        if (random.randint(0, 30) < 1):
            # Only space
            n_gen = random.randint(1, 60)
            for i in range(n_gen):
                generated += " "
        else:
            # Random length of string
            length = random.randint(1, max_length)

            while len(generated) < length:
                # Looping to add string
                if (random.randint(0,20) < 1):
                    # Add only space
                    n_gen = min(length - len(generated), random.randint(1, 2))
                    for i in range(n_gen):
                        generated += " "
                else:
                    # Add random latin chars
                    n_gen = min(length - len(generated), random.randint(0, 5))
                    generated += ''.join(random.choice(all_chars) for i in range(n_gen))

                    generated += " "

            if (length > len(generated) and random.randint(0, 20) < 3):
                # Add space to last
                n_gen = random.randint(0, length - len(generated))
                for i in range(n_gen):
                    generated += " "

        strings.append(generated)

    return fonts, strings
def random_sequences_sjnk(fonts):
    strings = []

    latin_chars = [x[:-1] for x in open("dicts/latin.txt", encoding="utf-8").readlines()]
    special_chars = [x[:-1] for x in open("dicts/special_char.txt", encoding="utf-8").readlines()]
    japan_chars = [x[:-1] for x in open("dicts/japan.txt", encoding="utf-8").readlines()]
    full_chars = latin_chars + special_chars + japan_chars
    font_dicts = {}
    max_length = 60
    print("full chars", len(full_chars))
    print(math.ceil(len(full_chars)*1.0 / max_length))
    # fonts = ['fonts/sjnk/Arial-Unicode-Regular.ttf'] * int(math.ceil(len(full_chars)*1.0 / max_length)) + fonts
    # for i in range(int(math.ceil(len(full_chars)*1.0 / max_length))):
    #     start_idx = max_length * i
    #     end_idx = min(max_length * (i+1), len(full_chars))
    #     strings.append("".join(full_chars[start_idx:end_idx]))
    #     # print("".join(full_chars[start_idx:end_idx]), len("".join(full_chars[start_idx:end_idx])))

    for font in fonts:
        if (font not in font_dicts):
            ttf = TTFont(font, fontNumber=0)

            chars = set([u'{0}'.format(chr(x[0])) for x in
                     list(chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables))])
            japan_chars_in_font = [x for x in japan_chars if check_character_in_font(x, ttf) and check_character_in_fontc1(x, font) and x in chars]
            latin_chars_in_font = [x for x in latin_chars if check_character_in_font(x, ttf) and check_character_in_fontc1(x, font) and x in chars]
            special_chars_in_font = [x for x in special_chars if check_character_in_font(x, ttf) and check_character_in_fontc1(x, font) and x in chars] + [" " for x in range(1,5)]

            font_dicts[font] = (japan_chars_in_font, latin_chars_in_font, special_chars_in_font)
        else:
            japan_chars_in_font, latin_chars_in_font, special_chars_in_font = font_dicts[font]

        generated = ""
        if (random.randint(0, 30) < 1):
            # Only space
            n_gen = random.randint(1, 60)
            for i in range(n_gen):
                generated += " "
        else:
            # if (random.randint(0, 20) < 3):
            #     # Add space to front
            #     n_gen = random.randint(0, 10)
            #     for i in range(n_gen):
            #         generated += " "

            # Random length of string
            length = random.randint(1, max_length)

            while len(generated) < length:
                # Looping to add string
                # if (random.randint(0,20) < 1):
                #     Add only space
                    # n_gen = min(length - len(generated), random.randint(0, 10))
                    # for i in range(n_gen):
                    #     generated += " "
                # else:
                if (random.randint(0, 10) < 3):
                    # Add random latin chars
                    n_gen = min(length - len(generated), random.randint(0, 5))
                    generated += ''.join(random.choice(latin_chars_in_font) for i in range(n_gen))
                else:
                    # Add random japanese chars
                    n_gen = min(length - len(generated), random.randint(0, 10))
                    generated += ''.join(random.choice(japan_chars_in_font) for i in range(n_gen))

                # Add a special char
                if (n_gen > 0 and len(generated) < length):

                    generated += random.choice(special_chars_in_font)

            if (length > len(generated) and random.randint(0, 20) < 3):
                # Add space to last
                n_gen = random.randint(0, length - len(generated))
                for i in range(n_gen):
                    generated += " "
        if (len(generated) == 0):
            n_gen = random.randint(1,10)
            generated += ''.join(random.choice(japan_chars_in_font) for i in range(n_gen))
        # if (len(generated) > 0 and generated[-1] == " "):
        #     generated = generated[:-1]
        strings.append(generated)
    return fonts, strings

def create_strings_randomly(length, allow_variable, count, let, num, sym, lang):
    """
        Create all strings by randomly sampling from a pool of characters.
    """

    # If none specified, use all three
    if True not in (let, num, sym):
        let, num, sym = True, True, True

    pool = ''
    if let:
        if lang == 'cn':
            pool += ''.join([chr(i) for i in range(19968, 40908)]) # Unicode range of CHK characters
        else:
            pool += string.ascii_letters
    if num:
        pool += "0123456789"
    if sym:
        pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"

    if lang == 'cn':
        min_seq_len = 1
        max_seq_len = 2
    else:
        min_seq_len = 2
        max_seq_len = 10

    strings = []
    for _ in range(0, count):
        current_string = ""
        for _ in range(0, random.randint(1, length) if allow_variable else length):
            seq_len = random.randint(min_seq_len, max_seq_len)
            current_string += ''.join([random.choice(pool) for _ in range(seq_len)])
            current_string += ' '
        strings.append(current_string[:-1])
    return strings

def print_text(file, list_):
    f = open(file, 'w', encoding="utf-8")

    f.writelines(list_)

def main():
    """
        Description: Main function
    """
    # Logging
    logging.basicConfig(filename='logs/logs_gen.txt',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    # Argument parsing
    args = parse_arguments()

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Creating word list
    lang_dict = load_dict(args.language)

    # Create font (path) list
    # fonts = load_fonts(args.language)
    fonts = load_fonts('vie')
    # print(fonts)
    # Creating synthetic sentences (or word)
    strings = []

    fonts_arr = [fonts[random.randrange(0, len(fonts))] for _ in range(0, args.count)]

    if args.use_wikipedia:
        strings = create_strings_from_wikipedia(args.length, args.count, args.language)
    elif args.input_file != '':
        strings = create_strings_from_file(args.input_file, args.count)
    elif args.random_sequences:
        strings = create_strings_randomly(args.length, args.random, args.count,
                                          args.include_letters, args.include_numbers, args.include_symbols, args.language)
        # Set a name format compatible with special characters automatically if they are used
        if args.include_symbols or True not in (args.include_letters, args.include_numbers, args.include_symbols):
            args.name_format = 2
    elif args.random_sequences_from_font:
        strings = create_strings_from_fonts(fonts_arr)
    elif args.random_sequences_sjnk:
        fonts_arr, strings = random_sequences_sjnk(fonts_arr)
    elif args.random_latin_sjnk:
        fonts_arr, strings = random_latin(fonts_arr)
    else:
        strings = create_strings_from_dict(args.length, args.random, args.count, lang_dict)


    string_count = len(strings)
    print("String count", string_count)
    print_text("src-train.txt", ['{}_{}.{}\n'.format(args.prefix, str(index), args.extension) for index in range(string_count)])
    print_text("tgt-train.txt", ['{}\n'.format(x) for x in strings])
    try:
        p = Pool(args.thread_count)
        p.starmap(
            FakeTextDataGenerator.generate,
            zip(
                [i for i in range(0, string_count)],
                strings,
                fonts_arr,
                [args.output_dir] * string_count,
                # [args.format] * string_count,
                [random.randint(args.format, args.format + 40) for x in range(string_count)],
                [args.extension] * string_count,
                [args.skew_angle] * string_count,
                [args.random_skew] * string_count,
                [args.blur] * string_count,
                [args.random_blur] * string_count,
                [args.background] * string_count,
                [args.distorsion] * string_count,
                [args.distorsion_orientation] * string_count,
                [args.handwritten] * string_count,
                [args.name_format] * string_count,
                [-1] * string_count,
                [args.prefix] * string_count
            )
        )
        p.terminate()
    except Exception as e:
        logging.error("Exception generate image ", exc_info=True)
        pass

    if args.name_format == 0:
        # Create file with filename-to-label connections
        with open(os.path.join(args.output_dir, "labels.txt"), 'w', encoding="utf8") as f:
            for i in range(string_count):
                file_name = str(i) + "." + args.extension
                f.write("{} {}\n".format(file_name, strings[i]))

if __name__ == '__main__':

    main()

