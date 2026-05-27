import pandas as pd


def load_data(client_path, property_path):

    clients = pd.read_csv(client_path)

    properties = pd.read_csv(property_path)

    return clients, properties