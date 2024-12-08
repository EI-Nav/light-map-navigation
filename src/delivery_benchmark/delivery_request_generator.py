import random
import argparse

def generate_delivery_instructions(num_instructions, instructions_per_block):
    # Define the relationship between buildings and units
    building_units = {
        "building1": ['unit1', 'unit2'],
        "building2": ['unit1'],
        "building3": ['unit1', 'unit2'],
        "building4": ['unit1', 'unit2'],
        "building5": ['unit1', 'unit2'],
        "building6": ['unit1', 'unit2'],
        "building7": ['unit1', 'unit2'],
        "building8": ['unit1', 'unit2', 'unit3'],
        "building9": ['unit1', 'unit2'],
        "building10": ['unit1', 'unit2', 'unit3'],
        "building11": ['unit1', 'unit2'],
        "building12": ['unit1', 'unit2', 'unit3'],
        "building13": ['unit1', 'unit2', 'unit3'],
        "building14": ['unit1', 'unit2', 'unit3'],
        "building15": ['unit1', 'unit2', 'unit3'],
        "building16": ['unit1', 'unit2', 'unit3'],
        "building17": ['unit1', 'unit2']
    }
    building_units_coords = {
        "building1": ['(-28.63, 29.34)', '(-24.94, 28.68)'],
        "building2": ['(-25.54, 9.27)'],
        "building3": ['(-34.89, -8.06)', '(-25.32, -7.53)'],
        "building4": ['(-34.69, -19.47)', '(-32.10, -20.92)'],
        "building5": ['(-24.44, -21.21)', '(-27.03, -19.76)'],
        "building6": ['(-29.96, -34.22)', '(-28.51, -31.63)'],
        "building7": ['(9.50, 25.45)', '(-6.49, 24.51)'],
        "building8": ['(-0.96, 7.53)', '(8.51, 7.97)', '(13.38, 7.57)'],
        "building9": ['(24.48, 8.08)', '(29.52, 8.06)'],
        "building10": ['(0.65, -15.58)', '(1.19, -12.14)', '(3.83, -16.40)'],
        "building11": ['(28.67, -17.74)', '(28.69, -10.10)'],
        "building12": ['(0.85, -27.57)', '(-2.59, -27.03)', '(1.66, -24.35)'],
        "building13": ['(60.60, 26.79)', '(64.26, 25.65)', '(57.59, 20.24)'],
        "building14": ['(90.63, 29.05)', '(89.51, 25.40)', '(84.05, 32.05)'],
        "building15": ['(77.66, 14.84)', '(77.10, 11.40)', '(74.47, 15.64)'],
        "building16": ['(96.04, 15.44)', '(94.59, 12.85)', '(97.15, 12.86)'],
        "building17": ['(90.41, -13.42)', '(89.47, -29.42)']
    }

    # Define a list of items that can be delivered
    boxes = ["box", "package", "parcel", "item"]

    instructions = []
    coordinates = []

    for _ in range(num_instructions):
        instruction = ""
        coordinate_block = []
        for _ in range(instructions_per_block):
            # Randomly select a box and a building
            selected_box = random.choice(boxes)
            selected_building = random.choice(list(building_units.keys()))

            # Randomly select a unit corresponding to the selected building
            selected_unit = random.choice(building_units[selected_building])

            # Determine the corresponding coordinates
            index = building_units[selected_building].index(selected_unit)
            selected_coords = building_units_coords[selected_building][index]

            # Generate the delivery instruction text
            description = f"Please deliver this {selected_box} into {selected_unit}, {selected_building}."
            instruction += " "
            instruction += description

            # Append the coordinates to the coordinate block
            coordinate_block.append(selected_coords)

        instructions.append(instruction)
        coordinates.append(" ".join(coordinate_block))  # Join coordinates with '; '

    return instructions, coordinates

def main(num_instructions, instructions_per_block, output_file, coords_file):
    # Generate the specified number of delivery instructions and organize them into blocks
    blocks, coordinates = generate_delivery_instructions(num_instructions, instructions_per_block)

    # Write the generated instructions to a text file, with each block on a single line
    with open(output_file, "w") as file:
        for idx in range(len(blocks)):
            file.write(blocks[idx] + "\n")

    # Write the corresponding coordinates to another file, each line corresponds to the instructions
    with open(coords_file, "w") as file:
        for idx in range(len(coordinates)):
            file.write(coordinates[idx] + "\n")

    print(f"{num_instructions} delivery instructions in blocks of {instructions_per_block} written to '{output_file}'.")
    print(f"Coordinates for each block written to '{coords_file}'.")

if __name__ == "__main__":
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate delivery instructions and corresponding coordinates, and write them to files.")
    parser.add_argument("--num", type=int, required=True, help="Total number of instructions to generate")
    parser.add_argument("--per-block", type=int, required=True, help="Number of instructions per block")
    parser.add_argument("--output-file", type=str, required=True, help="Name of the output file for instructions")
    parser.add_argument("--coords-file", type=str, required=True, help="Name of the output file for coordinates")

    args = parser.parse_args()
    main(args.num, args.per_block, args.output_file, args.coords_file)
