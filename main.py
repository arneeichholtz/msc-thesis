from features_config.features import *

def phoneme_processing():
    # Create map from label to label index for each feature
    phonation_map = {label: i for i, label in enumerate(phonation_labels.keys())}
    manner_map    = {label: i for i, label in enumerate(manner_labels.keys())}
    place_map     = {label: i for i, label in enumerate(place_labels.keys())}
    fb_map        = {label: i for i, label in enumerate(fb_labels.keys())}
    round_map     = {label: i for i, label in enumerate(round_labels.keys())}
    central_map   = {label: i for i, label in enumerate(central_labels.keys())}

    # Construct the final phoneme_dict, this contains all the 59 phonemes in the MV system, with their corresponding feature values as a list of indices
    full_phoneme_feature_dict = {
        phoneme: [
            phonation_map[feature_values[0]],
            manner_map[feature_values[1]],
            place_map[feature_values[2]],
            fb_map[feature_values[3]],
            round_map[feature_values[4]],
            central_map[feature_values[5]]
        ]
        for phoneme, feature_values in mv_data.items()
    }

    # Maps all phoneme labels to the 39+1 labels that will be used 
    phoneme_feature_dict = {
        phoneme_mapping[phoneme]: 
            features for phoneme, features in full_phoneme_feature_dict.items()
    }

    return full_phoneme_feature_dict, phoneme_feature_dict

if __name__ == "__main__":

    _, phoneme_feature_dict = phoneme_processing()

    print("Number of phonemes:", len(phoneme_feature_dict))
    
    # print(phoneme_feature_dict["em"])
    
    for phoneme, features in phoneme_feature_dict.items():
        print(f"{phoneme}: {features}")




# Steps in code
# Load wav2vec2 model
# Load dataset
# Convert phoneme transcriptions to articulatory feature (AF) vectors
# What is the frame length for wav2vec2? Use this to determine a label for each audio frame
# Write code to add linear projection on top of wav2vec2 to predict AF vectors for each frame
# Implement training pipeline to calculate MSE between the k=6 AF ground-truth vector and predicted AF vector for each frame.


# Implementation questions:
# - How many layers of wav2vec do we fine-tune? 
# - Can I find a fine-tune script from wav2vec2 or CBM paper?
# - Understand fine-tune details from wav2vec2 and also CBM paper.