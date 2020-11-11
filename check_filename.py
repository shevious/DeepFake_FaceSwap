import os
def find_all_files(directory):
    for root, dirs, files in os.walk(directory):

        for file in files:
            p=os.path.join(root, file)
            p=p.split("/")[len(p.split("/"))-2]
            name, ext = os.path.splitext(p)

            yield os.path.join(root, file)


input_directory = 'nic'
folder_img = find_all_files(input_directory)
for filename in folder_img:
    for c in filename:
        print(c)
    print(filename)

input_directory = 'chris'
folder_img = find_all_files(input_directory)
for filename in folder_img:
    for c in filename:
        print(c)
    print(filename)
