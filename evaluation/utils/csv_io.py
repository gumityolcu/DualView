import torch
import numpy as np
import matplotlib.pyplot as plt
import csv


# CSV FILE INTERACTION

def write_data(X, Y, file_name):
    X = X.cpu()
    Y = Y.cpu()
    data = X.detach().numpy()
    labels = Y.detach().numpy()
    with open(file_name, "w") as file:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                file.write(f"{float(data[i][j])}, ")
            file.write(f"{labels[i]}")
            if i != X.shape[0]:
                file.write("\n")
    return data, labels


def write_matrix(X, file_name):
    X = X.cpu()
    data = X.detach().numpy()
    with open(file_name, "w") as file:
        for i in range(x.shape[0]):
            for j in range(X.shape[1]):
                file.write(f"{float(data[i][j])}")
                if j != X.shape[1] - 1:
                    file.write(",")
            if j != X.shape[0] - 1:
                file.write(f"\n")
    return data


def write_3D_tensor(X, file_name):
    X = X.cpu()
    data = X.detach().numpy()
    with open(file_name, "w") as file:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    file.write(f"{float(data[i][j][k])}")
                    if k != X.shape[2] - 1:
                        file.write("!")
                if j != X.shape[1] - 1:
                    file.write(",")
            if i != X.shape[0] - 1:
                file.write(f"\n")
    return data


def read_data(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        data = []
        labels = []
        for row in reader:
            list = []
            for i, entry in enumerate(row):
                if i < len(row) - 1:
                    list.append(float(entry.strip()))
                else:
                    labels.append(int(entry.strip()))
            data.append(list)
        data = np.array(data)
        labels = np.array(labels)
        return data, labels


def read_matrix(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        mat = []
        for row in reader:
            list = []
            for i, entry in enumerate(row):
                list.append(float(entry.strip()))
            mat.append(list)
        mat = np.array(mat)
        return mat


def read_3D_tensor(file_name):
    with open(file_name, newline='') as csvfile:
        reader0 = csv.reader(csvfile, delimiter=',', quotechar='|')
        mat = []
        for dim1 in reader0:
            list1 = []
            for dim2 in dim1:
                dim3 = dim2.split("!")
                list2 = []
                for i, entry in enumerate(dim3):
                    list2.append(float(entry.strip()))
                list1.append(list2)
            mat.append(list1)
        mat = np.array(mat)
        return mat
