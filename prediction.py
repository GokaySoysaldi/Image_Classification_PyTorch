#from deep_learning_model.training.models import Classifier
#from deep_learning_model.training.config import CLASSES, MODEL_NAME
from model.model import CNN

import torch
import torchvision.transforms as transforms

import os



class ImageClassifier:
    def __init__(self):        
        self.classifier = CNN()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join('model', 'saved_model.pth')                 
        self.classifier.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.classifier.eval()
        self.CLASSES = ['plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def predict(self, image):            
        transforms_image = transforms.Compose(
            [transforms.Resize((32, 32)),
            transforms.ToTensor()])        
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = transforms_image(image) 
        image = image.unsqueeze(0)
        image = image.to(self.device)

        output = self.classifier(image)
        class_idx = torch.argmax(output, dim=1)
        output = output.tolist()   
        # print(type(output[0][1]))
        return [self.CLASSES[class_idx],max(output[0])]