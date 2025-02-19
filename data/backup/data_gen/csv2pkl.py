import pickle

def read_csv_file(csv_file):
    data = {}
    with open(csv_file, 'r') as file:
        for line in file:
            rows = line.strip().split()
            if len(rows) >= 2: 
                data[rows[0]] = str(rows[1])
    return data

def save_as_pickle(data, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)

def main():
    csv_file = '../data_csv/label.csv'  
    pkl_file = '../data_pkl/label.pkl' 

    labels = read_csv_file(csv_file)
    save_as_pickle(labels, pkl_file)
    print(f"Labels saved to {pkl_file}")

if __name__ == "__main__":
    main()
