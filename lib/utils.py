import os
import warnings
import logging

import numpy as np
import torch
from torchvision import datasets

from lib.gaussian.gaussianimage_cholesky_batch import GaussianImage_Cholesky_Batch


warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger("lib.utils")


def get_initialized_gs_batch(num_classes, batch_size, gs_dir, gs_type, gaussian_cfg, epochs, device):
    if epochs is None:
        epochs = "final"
        
    sample_index_path = os.path.join(gs_dir, f"ipc_samples_0_{num_classes}.npy")
    sample_index = np.load(sample_index_path, allow_pickle=True)
    ckpt_path = os.path.join(gs_dir, "mixed", f"batch_0-{batch_size-1}", f"model_{epochs}.pth")

    if gs_type == "GaussianImage_Cholesky_Batch":
        gs_model = GaussianImage_Cholesky_Batch(gaussian_cfg, device)
    else:
        raise ValueError(f"Unknown gs_type: {gs_type}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_dict = gs_model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    gs_model.load_state_dict(model_dict)
    return gs_model, sample_index


def load_gs_model(args):
    if os.path.isfile(args.load_path):
        checkpoint_path = args.load_path
    else:
        checkpoint_name = f"GSDD_TM_{args.ipc}ipc_iter{args.iteration}.pt"
        checkpoint_path = os.path.join(args.load_path, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    
    
    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    gs_models, syn_lr = ckpt['model'], ckpt['syn_lr']
    args.lr_net = syn_lr.item()

    print(f"Loading synthetic lr as {args.lr_net}")

    # Set the model to evaluation mode
    gs_models.eval()
    return gs_models


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
    dict = {
        "imagenette" : imagenette,
        "imagewoof" : imagewoof,
        "imagefruit": fruits,
        "imageyellow": yellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
        "geo": imagegeo,
        "food": imagefood,
        "mammals": imagemammal,
        "marine": marine,
        "a": alpha,
        "b": beta,
        "c": gamma,
        "d": delta,
        "e": epsilon
    }

    mean = torch.tensor([0.4377, 0.4438, 0.4728]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


config = Config()


def get_dataset_for_init(dataset_name, dataset_path, subset=None, resolution=None, transform=None, lazy=False):
    dataset_name = dataset_name.lower()
    assert dataset_name in ["mnist", "cifar10", "cifar100", "imagenet"], f"Dataset {dataset_name} not supported"
    if dataset_name == "imagenet" or dataset_name == "imagenet":
        assert subset in config.dict.keys(), f"Subset {subset} not supported"
    
    if dataset_name == "mnist":
        dataset_cls = datasets.MNIST
        num_classes = 10
        img_resolution = (28, 28)
    elif dataset_name == "cifar10":
        dataset_cls = datasets.CIFAR10
        num_classes = 10 
        img_resolution = (32, 32)
    elif dataset_name == "cifar100":
        dataset_cls = datasets.CIFAR100
        num_classes = 100
        img_resolution = (32, 32)
    elif dataset_name == "imagenet":
        dataset_cls = datasets.ImageFolder
        if subset is None:
            num_classes = 1000
            img_resolution = (224, 224)
        else:
            num_classes = len(config.dict[subset])
            img_resolution = (128, 128)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))
    
    if resolution is not None:
        img_resolution = (resolution, resolution)
    
    if dataset_name == "imagenet":
        if subset in ["imagenette", "imagewoof", "imagefruit", "imageyellow", "imagemeow", "imagesquawk"]:
            train_dataset = dataset_cls(os.path.join(dataset_path, "train"), transform=transform)
            test_dataset = dataset_cls(os.path.join(dataset_path, "val"), transform=transform)
        elif subset is None:
            train_dataset = dataset_cls(os.path.join(dataset_path, "train"), transform=transform)
            test_dataset = dataset_cls(os.path.join(dataset_path, "val"), transform=transform)
        else:
            raise ValueError("Invalid subset for imagenet")
    else:
        train_dataset = dataset_cls(dataset_path, train=True, download=False, transform=transform)
        test_dataset = dataset_cls(dataset_path, train=False, download=False, transform=transform)

    return train_dataset, test_dataset, num_classes, img_resolution
