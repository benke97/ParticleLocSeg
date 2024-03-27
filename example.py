from models import localization_UNet, segmentation_UNet
from helpers import preprocess_image, center_of_mass_positions
import torch
import matplotlib.pyplot as plt


# Prepare the image
image_path = 'test_images/03.tif'
image = preprocess_image(image_path)


# Load the models
localization_model = localization_UNet()
model_weights_path = 'model_weights/weights_localization_model.pth'
state_dict = torch.load(model_weights_path)['model_state_dict']
localization_model.load_state_dict(state_dict)
localization_model.eval()

segmentation_model = segmentation_UNet()
model_weights_path = 'model_weights/weights_segmentation_model.pth'
state_dict = torch.load(model_weights_path)['model_state_dict']
segmentation_model.load_state_dict(state_dict)
segmentation_model.eval()


# Find columns and label them
with torch.no_grad():
    output = localization_model(image)
    predicted_positions = center_of_mass_positions(output)

    segmentation_input = torch.where(output >= 0.1, 1.0, 0.0) # Threshold the localization output at 0.1
    segmentation_output = segmentation_model(segmentation_input)
    binarized_output = torch.where(segmentation_output >= 0.5, 1.0, 0.0) # Make output binary
    predicted_labels = [binarized_output[0,0,int(pos[0]+0.5),int(pos[1]+0.5)].cpu().numpy() for pos in predicted_positions] # Sample the pixel value of the predicted positions in the segmentation output to get label


# Display the image with predicted positions colored by label
colors = ['springgreen' if label == 1 else 'darkorange' for label in predicted_labels]
save_name = 'predicted_positions.png'
plt.figure(figsize=(6,6))
plt.imshow(image.squeeze(), cmap='gray')
plt.scatter(predicted_positions[:, 1], predicted_positions[:, 0], c=colors, s=40)
plt.axis('off')
plt.tight_layout()
plt.savefig('figures/' + save_name, dpi=300)
plt.show()