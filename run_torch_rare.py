
import os
import cv2
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from features_extractor import FeatureExtractor
from torch_rare import DeepRare


def load_model(model_name: str):
    """
    Charge un modèle CNN PyTorch.

    Args:
        model_name (str): Nom du modèle pré-entraîné à charger (ex: 'resnet18', 'vgg16').
        custom_model_path (str, optional): Chemin vers un modèle personnalisé enregistré.

    Returns:
        torch.nn.Module: Modèle chargé.
    """


    print(f"Chargement du modèle pré-entraîné : {model_name}")
    if model_name.lower() == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name.lower() == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError(f"Modèle non supporté : {model_name}")

    model.eval()  # Mettre le modèle en mode évaluation
    return model


def parse_list_of_ints(value: str):
    """
    Parse une chaîne de caractères en liste d'entiers.

    Args:
        value (str): Chaîne de caractères à parser, format attendu: "1,2,3"

    Returns:
        list[int]: Liste d'entiers.
    """
    try:
        return [int(x) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' n'est pas une liste d'entiers valide. Format attendu: '1,2,3'")


def process_image(image):
        # Load an example image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(img).unsqueeze(0)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Charge un modèle CNN en PyTorch.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="mobilenet_v2", 
        help="Nom du modèle pré-entraîné à charger (ex: resnet18, vgg16, mobilenet_v2)."
    )

    parser.add_argument(
        "--layers_to_extract", 
        type=parse_list_of_ints, 
        default="1,2,  4,5,8,  9,11,12,13,  16,17,18,19,  26,27,28,29", 
        help="Liste d'entiers séparés par des virgules (ex: 1,2,3)."
    )

    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )


    args = parser.parse_args()

    try:
        model = load_model(args.model_name)
        print(f"Modèle chargé avec succès : {args.model_name}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")


    rarity_network = DeepRare(threshold=args.threshold) # instantiate class
    directory = r'input'

    for filename in os.listdir(directory):
        print(filename)
        go_path = os.path.join(directory, filename)

        img = cv2.imread(go_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        input_image = process_image(img)

        # Create a feature extractor instance
        feature_extractor = FeatureExtractor(model, args.layers_to_extract)

        # Get feature maps
        with torch.no_grad():  # Disable gradient computation for inference
            feature_maps = feature_extractor(input_image)

        print(f"Numbers features maps : {len(feature_maps)}")

        for layer in feature_maps:
            print(f" - {layer.shape}")

        SAL, groups = rarity_network(feature_maps)


        plt.figure(1)

        plt.subplot(421)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Initial Image')

        plt.subplot(422)
        plt.imshow(SAL)
        plt.axis('off')
        plt.title('Final Saliency Map')

        for i in range(0,groups.shape[-1]):

            plt.subplot(423 + i)
            plt.imshow(groups[:, :, i])
            plt.axis('off')
            plt.title(f'Level {i}Saliency Map')

        plt.show()