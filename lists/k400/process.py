import os

# path = "/home/linrui/moe/MoTE/lists/k400/testlist.txt"

def new_list(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split("/")[1] for line in lines]
        # print(lines[:5])

    with open(path.split(".")[0]+"_new.txt", "w") as f:
        for line in lines:
            f.write(line + "\n")

if __name__ == "__main__":
    base_path = "/home/linrui/moe/MoTE/lists/k400"
    for split in ["trainlist", "vallist", "testlist"]:
        path = os.path.join(base_path, f"{split}.txt")
        new_list(path)