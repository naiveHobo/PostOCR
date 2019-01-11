
def get_distance_matrix(orig, edited):
    # initialize the matrix
    orig_len = len(orig) + 1
    edit_len = len(edited) + 1
    distance_matrix = [[0] * edit_len for _ in range(orig_len)]
    for i in range(orig_len):
        distance_matrix[i][0] = i
    for j in range(edit_len):
        distance_matrix[0][j] = j

    # calculate the edit distances
    for i in range(1, orig_len):
        for j in range(1, edit_len):

            deletion = distance_matrix[i - 1][j] + 1
            insertion = distance_matrix[i][j - 1] + 1
            substitution = distance_matrix[i - 1][j - 1]

            if orig[i - 1] != edited[j - 1]:
                substitution += 1

            distance_matrix[i][j] = min(insertion, deletion, substitution)

    return distance_matrix


class Compare:
    def __init__(self, original, edited):
        self.original = original
        self.edited = edited

        self.distance_matrix = get_distance_matrix(original, edited)
        i = len(self.distance_matrix) - 1
        j = len(self.distance_matrix[i]) - 1
        self.edit_distance = self.distance_matrix[i][j]
        self.num_orig_elements = i

    def __repr__(self):
        edited_str = str(self.edited)
        original_str = str(self.original)
        if len(edited_str) > 10:
            edited_str = edited_str[10:] + " ..."

        if len(original_str) > 10:
            original_str = original_str[10:] + " ..."
        return "Compare({}, {})".format(edited_str, original_str)

    def set_alignment_strings(self):
        original = self.original
        edited = self.edited
        num_orig_elements = self.num_orig_elements
        i = num_orig_elements
        j = len(self.edited)

        # edit_distance = self.edit_distance
        distance_matrix = self.distance_matrix

        num_deletions = 0
        num_insertions = 0
        num_substitutions = 0

        align_orig_elements = []
        align_edited_elements = []
        align_label_str = []

        # start at the cell containing the edit distance and analyze the
        # matrix to figure out what is a deletion, insertion, or
        # substitution.
        while i or j:
            # if deletion
            if distance_matrix[i][j] == distance_matrix[i - 1][j] + 1:
                num_deletions += 1

                align_orig_elements.append(original[i - 1])
                align_edited_elements.append(" ")
                align_label_str.append('D')

                i -= 1

            # if insertion
            elif distance_matrix[i][j] == distance_matrix[i][j - 1] + 1:
                num_insertions += 1

                align_orig_elements.append(" ")
                align_edited_elements.append(edited[j - 1])
                align_label_str.append('I')

                j -= 1

            # if match or substitution
            else:
                orig_element = original[i - 1]
                edited_element = edited[j - 1]

                if orig_element != edited_element:
                    num_substitutions += 1
                    label = 'S'
                else:
                    label = ' '

                align_orig_elements.append(orig_element)
                align_edited_elements.append(edited_element)
                align_label_str.append(label)

                i -= 1
                j -= 1

        align_orig_elements.reverse()
        align_edited_elements.reverse()
        align_label_str.reverse()

        self.align_orig_elements = align_orig_elements
        self.align_edited_elements = align_edited_elements
        self.align_label_str = align_label_str

    def show_changes(self):
        if not hasattr(self, 'align_orig_elements'):
            self.set_alignment_strings()

        assert (len(self.align_orig_elements) ==
                len(self.align_edited_elements) ==
                len(self.align_label_str))

        assert len(self.align_label_str) == len(self.align_edited_elements) == len(
            self.align_orig_elements), "different number of elements"

        # for each word in line, determine whether there's a change and append with the according format
        print_string = ''
        for index in range(len(self.align_label_str)):
            if self.align_label_str[index] == ' ':
                print_string += self.align_edited_elements[index] + ' '
            elif self.align_label_str[index] == 'S' or self.align_label_str[index] == 'I':
                element = self.align_edited_elements[index]
                print_string += changed(element) + ' '
            else:  # a deletion - need to print what was in the original that got deleted
                element = self.align_orig_elements[index]
                print_string += changed(element)
        return print_string


def changed(plain_text):
    return "<CORRECTION>" + plain_text + "</CORRECTION>"
