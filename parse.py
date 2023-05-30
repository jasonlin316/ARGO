def parse_txt_file(filename):
    data = {}

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line and ':' and '=' in line:
                line = line[line.index(': ')+1:]  # Remove everything before and including ":"
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()

                if key in data:
                    data[key].append(value)
                else:
                    data[key] = [value]

    return data

def print_lists(data):
    for key, values in data.items():
        print(f"{key}: {values}")


# Replace 'input.txt' with the path to your input file
filename = 'thread_exp_output.txt'
data = parse_txt_file(filename)
print_lists(data)
