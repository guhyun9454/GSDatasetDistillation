import os
import argparse
from tqdm import tqdm


class Config:
    custom = [1, 199, 388, 294, 340, 932, 327, 765, 928, 486]
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]
    # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]
    # ["rock_beauty", "clownfish", "loggerhead", "puffer", "stingray", "jellyfish", "starfish", "eel", "anemone", "american_lobster"]
    imageblub = [392, 393, 33, 397, 6, 107, 327, 390, 108, 122]
    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]
    alyosha = [292, 340, 971, 987, 130, 323, 937, 337, 199, 294]
    # [scotty, brown bear, beaver, husky, bee, panther, terrapin, tiger, badger, duck
    mascots = [199, 294, 337, 250, 309, 286, 36, 292, 362, 97]
    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    fruits = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]
    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    yellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]
    # ["baseball", "basketball", "croquet_ball", "golf_ball", "ping-pong_ball", "rugby_ball", "soccer_ball", "tennis_ball", "volleyball", "puck"]
    imagesport = [429, 430, 522, 574, 722, 768, 805, 852, 890, 746]
    # ["saxophone", "trumpet", "french_horn", "flute", "oboe", "ocarina", "bassoon", "trombone", "panpipe", "harmonica"]
    imagewind = [776, 513, 566, 558, 683, 684, 432, 875, 699, 593]
    # ["saxophone", "trumpet", "french_horn", "flute", "oboe", "ocarina", "bassoon", "trombone", "panpipe", "harmonica"]
    imagestrings = [776, 513, 566, 558, 683, 684, 432, 875, 699, 593]
    # ["volcano", "alp", "lakeside", "geyser", "coral_reef", "sandbar", "promontory", "seashore", "cliff", "valley"]
    imagegeo = [980, 970, 975, 974, 973, 977, 976, 978, 972, 979]
    # ["axolotl", "tree_frog", "king_snake", "african_chameleon", "iguana", "eft", "fire_salamander", "box_turtle", "american_alligator", "agama"]
    imageherp = [29, 31, 56, 47, 39, 27, 25, 37, 50, 42]
    # ["cheeseburger", "hotdog", "pretzel", "pizza", "french loaf", "icecream", "guacamole", "carbonara", "bagel", "trifle"]
    imagefood = [933, 934, 932, 963, 930, 928, 924, 959, 931, 927]
    # ["fire_engine", "garbage_truck", "forklift", "racer", "tractor", "unicycle", "rickshaw", "steam_locomotive", "bullet_train", "mountain_bike"]
    imagewheels = [555, 569, 561, 751, 866, 880, 612, 820, 466, 671]
    # ["bubble", "piggy_bank", "stoplight", "coil", "kimono", "cello", "combination_lock", "triumphal_arch", "fountain", "cowboy_boot"]
    imagemisc = [971, 719, 920, 506, 614, 486, 507, 873, 562, 514]
    # ["broccoli", "cauliflower", "mushroom", "cabbage", "cardoon", "mashed_potato", "artichoke", "corn", "fountain", "spaghetti_squash"]
    imageveg = [971, 719, 920, 506, 614, 486, 507, 873, 562, 940]
    # ["ladybug", "bee", "monarch", "dragonfly", "mantis", "black_widow", "rhinoceros_beetle", "walking_Stick", "grasshopper", "scorpion"]
    imagebug = [301, 309, 323, 319, 315, 75, 306, 313, 311, 71]
    # ["african_elephant", "red_panda", "camel", "zebra", "guinea_pig", "kangaroo", "platypus", "arctic_fox", "porcupine", "gorilla"]
    imagemammal = [386, 387, 354, 340, 338, 104, 103, 279, 334, 366]
    # ["orca", "great_white_shark", "puffer", "starfish", "loggerhead", "sea lion", "jellyfish", "anemone", "rock crab", "rock beauty"]
    marine = [148, 2, 397, 327, 33, 150, 107, 108, 119, 392]

    # ['Leonberg', 'proboscis monkey, Nasalis larvatus', 'rapeseed', 'three-toed sloth, ai, Bradypus tridactylus', 'cliff dwelling', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'hamster', 'gondola', 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'limpkin, Aramus pictus']
    alpha = [255, 376, 984, 364, 500, 986, 333, 576, 148, 135]
    # ['spoonbill', 'web site, website, internet site, site', 'lorikeet', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'earthstar', 'trolleybus, trolley coach, trackless trolley', 'echidna, spiny anteater, anteater', 'Pomeranian', 'odometer, hodometer, mileometer, milometer', 'ruddy turnstone, Arenaria interpres']
    beta = [129, 916,  90, 275, 995, 874, 102, 259, 685, 139]
    # ['freight car', 'hummingbird', 'fireboat', 'disk brake, disc brake', 'bee eater', 'rock beauty, Holocanthus tricolor', 'lion, king of beasts, Panthera leo', 'European gallinule, Porphyrio porphyrio', 'cabbage butterfly', 'goldfinch, Carduelis carduelis']
    gamma = [565,  94, 554, 535,  92, 392, 291, 136, 324,  11]
    # ['ostrich, Struthio camelus', 'Samoyed, Samoyede', 'junco, snowbird', 'Brabancon griffon', 'chickadee', 'sorrel', 'admiral', 'great grey owl, great gray owl, Strix nebulosa', 'hornbill', 'ringlet, ringlet butterfly']
    delta = [  9, 258,  13, 262,  19, 339, 321,  24,  93, 322]
    # ['spindle', 'toucan', 'black swan, Cygnus atratus', 'king penguin, Aptenodytes patagonica', "potter's wheel", 'photocopier', 'screw', 'tarantula', 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'lycaenid, lycaenid butterfly']
    epsilon = [816,  96, 100, 145, 739, 713, 783,  76, 688, 326]
    # dict = {
    #     "imagenette" : imagenette,
    #     "woof" : imagewoof,
    #     "imagefruit": fruits,
    #     "yellow": yellow,
    #     "cats": imagemeow,
    #     "birds": imagesquawk,
    #     "geo": imagegeo,
    #     "food": imagefood,
    #     "mammals": imagemammal,
    #     "marine": marine,
    #     "a": alpha,
    #     "b": beta,
    #     "c": gamma,
    #     "d": delta,
    #     "e": epsilon
    # }

    dict = {
        "imagenette" : imagenette,
        "imagewoof" : imagewoof,
        "imagefruit": fruits,
        "imageyellow": yellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
    }

def create_symlinks(imagenet_path, output_path):
    """
    Creates directories for ImageNet subsets and populates them with
    relative symbolic links to the original ImageNet data.

    Args:
        imagenet_path (str): The path to the root of the original
                             ImageNet dataset. This directory should
                             contain 'train' and 'val' folders.
        output_path (str): The path where the new subset directories
                           will be created.
    """
    config = Config()

    # --- 1. Get the sorted list of ImageNet class folder names ---
    train_path = os.path.join(imagenet_path, 'train')
    if not os.path.isdir(train_path):
        print(f"Error: 'train' directory not found in '{imagenet_path}'")
        return

    all_class_folders = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    print(f"Found {len(all_class_folders)} classes in {train_path}")

    for subset_name, class_indices in tqdm(config.dict.items(), desc="Processing subsets"):
        print(f"\nCreating dataset for: '{subset_name}'")

        subset_dir = os.path.join(output_path, subset_name)

        train_subset_dir = os.path.join(subset_dir, 'train')
        val_subset_dir = os.path.join(subset_dir, 'val')

        os.makedirs(train_subset_dir, exist_ok=True)
        os.makedirs(val_subset_dir, exist_ok=True)

        for index in tqdm(class_indices, desc=f"Linking '{subset_name}' classes", leave=False):
            if index >= len(all_class_folders):
                print(f"Warning: Index {index} is out of bounds for subset '{subset_name}'. Skipping.")
                continue

            class_folder_name = all_class_folders[index]

            src_train = os.path.join(imagenet_path, 'train', class_folder_name)
            dst_train = os.path.join(train_subset_dir, class_folder_name)
            
            link_src_train = os.path.relpath(src_train, train_subset_dir)

            if not os.path.lexists(dst_train):
                os.symlink(link_src_train, dst_train)

            src_val = os.path.join(imagenet_path, 'val', class_folder_name)
            dst_val = os.path.join(val_subset_dir, class_folder_name)
            
            if os.path.isdir(src_val):
                link_src_val = os.path.relpath(src_val, val_subset_dir)
                if not os.path.lexists(dst_val):
                    os.symlink(link_src_val, dst_val)

    print("All symbolic links created successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create ImageNet subset directories with relative symbolic links.")

    parser.add_argument(
        '--imagenet_path',
        type=str,
        required=True,
        help="The path to the root ImageNet directory (containing train/val folders)."
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="The path where the new subset directories (e.g., 'imagenette', 'imagewoof') will be created."
    )

    args = parser.parse_args()

    create_symlinks(args.imagenet_path, args.output_path)