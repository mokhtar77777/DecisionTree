def tabs_generator(num_of_tabs):
    tabs = ""
    for tab_num in range(num_of_tabs):
        tabs += '\t'

    return tabs


def write_line(filename, line, depth, cur_class):
    tabs = tabs_generator(num_of_tabs=depth)

    if cur_class is None:
        line_to_be_written = tabs + "prediction = " + line + '\n'

    else:
        if cur_class:
            line_to_be_written = tabs + "if " + line + ":\n"

        else:
            line_to_be_written = tabs + "else:\n"

    if filename is None:
        print(line_to_be_written)

    else:
        with open(filename, 'a') as file:
            file.write(line_to_be_written)
            file.flush()
