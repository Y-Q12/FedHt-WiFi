import svhn2mnist
import usps
import syn2gtrsb

def Generator(source, target, pixelda=False):
    return svhn2mnist.Feature()

def Disentangler():
    return svhn2mnist.Feature_disentangle()

def Classifier(source, target):
    return svhn2mnist.Predictor()
def Feature_Discriminator(input_dim=2048):
    return svhn2mnist.Feature_discriminator(input_dim)

def Reconstructor():
    return svhn2mnist.Reconstructor()

def Mine():
    return svhn2mnist.Mine()

#