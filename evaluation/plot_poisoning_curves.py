import json
import torch
import matplotlib.pyplot as plt
import os

methods={
    "CIFAR":[('dualview','blue','DV'), ('influence', 'green', 'IF'), ('representer', 'c', 'RP'),('rp_similarity','purple','SIM-rp')],
    "MNIST":[ ('dualview','blue','DV'), ('influence', 'green', 'IF'), ('representer', 'c', 'RP'),('rp_similarity','purple','SIM-rp'),('similarity','black','SIM')]
}
CIFAR_dir="/home/fe/yolcu/Documents/Code/THESIS/results/CIFAR/resnet/corrupt/"
MNIST_dir="/home/fe/yolcu/Documents/Code/THESIS/results/MNIST/basic_conv/corrupt/"

for p,t in [(CIFAR_dir, "CIFAR"),(MNIST_dir,"MNIST")]:
    plt.figure(figsize=(6,4))
    plt.xlabel('Ratio of Controlled Images')
    plt.ylabel('Ratio of Detected Poisoned Samples')
    for m,c,n in methods[t]:
        with open(f"{p}{t}_corrupt_{m}_eval_results.json") as file:
            res = json.load(file)
        arr=res['label_flipping_curve']
        x=torch.range(1,len(arr))/len(arr)
        plt.plot(x,arr,c,label=n)
    plt.plot([0.,1.],[0.,1.],linestyle="dashed",color="black",label="RAND")
    plt.plot([0.,res['num_corrupt_samples']/len(arr),1.],[0.,1.,1.],color="gray", linestyle="dashed")
    plt.plot([0.,1-res['num_corrupt_samples']/len(arr),1.],[0.,0.,1.],color="gray", linestyle="dashed")
    _,x_lim=plt.xlim()
    _,y_lim=plt.ylim()
    plt.xlim((0.,x_lim))
    plt.ylim((0.,y_lim))
    plt.legend(loc="lower right")
    plt.savefig(f"{p}{t}_label_posioning_curve.pdf")
    os.system(f"pdfcrop {p}{t}_label_posioning_curve.pdf")
    os.system(f"rm {p}{t}_label_posioning_curve.pdf")
    os.system(f"mv {p}{t}_label_posioning_curve-crop.pdf {p}{t}_label_posioning_curve.pdf")
