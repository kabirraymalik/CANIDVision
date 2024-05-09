import os
import torch
from models import VideoClassifier
import matplotlib.pyplot as plt

class learningUtils():
    def __init__(self):
        self.saveCount = 0
        self.savePath = ''

    def _save_model(self, model):
        if self.saveCount == 0:
            self.savePath = os.curdir+'/models/session_'+str(len(os.listdir(os.curdir+'/models/'))+1)+'/'
            os.mkdir(self.savePath)
        self.saveCount += 1
        saveTo = self.savePath+'save_'+str(self.saveCount)
        torch.save(model.state_dict(), saveTo)

    def _save_stats(self, accuracies, losses):
        numSteps = len(losses)/len(accuracies)
        stepCount = 0
        sum = 0
        avgLoss = []
        for lossVal in losses:
            if len(avgLoss) == 0:
                avgLoss.append(lossVal)
            if stepCount == numSteps:
                avgLoss.append(sum/numSteps)
                stepCount = 0
                sum = 0
            sum += lossVal
            stepCount += 1
        figure, axis = plt.subplots(2) 
        figure.suptitle("accuracy and avg error vs epoch")
        axis[0].plot(accuracies)
        axis[0].set_ylim([0,100])
        axis[0].set_title("accuracy vs time") 
        axis[1].plot(avgLoss)
        axis[1].set_ylim([0,max(avgLoss)])
        axis[1].set_title("avg. loss vs time") 
        path = os.curdir+'/data/testdata.png'
        figure.savefig(path)