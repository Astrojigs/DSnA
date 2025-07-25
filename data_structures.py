class ListNode:
    def __init__(self, data=None):
        self.data = data
        self.next = None


class LinkedList:
    """
    Create a Linked List.

    methods:
        append:
            add an element at the end of the list.
        length:
            returns the length of the list
    """

    def __init__(self):
        self.head = ListNode()

    def append(self, data):
        # Creating a new node with the data
        new_node = ListNode(data)

        cur = self.head
        while cur.next is not None:
            cur = cur.next

        cur.next = new_node

    def length(self):
        cur = self.head
        total = 0
        while cur.next is not None:
            total += 1
            cur = cur.next
        return total

    def display(self):
        elements = []
        cur_node = self.head
        while cur_node.next is not None:
            cur_node = cur_node.next
            elements.append(cur_node.data)
        print(elements)

    def __repr__(self):
        elements = []
        cur_node = self.head
        while cur_node.next is not None:
            cur_node = cur_node.next
            elements.append(cur_node.data)
        return elements

    def __str__(self):
        elements = []
        cur_node = self.head
        while cur_node.next is not None:
            cur_node = cur_node.next
            elements.append(cur_node.data)
        return f'{elements}, <LinkedList obj>'

    def __len__(self):
        elements = []
        cur_node = self.head
        while cur_node.next is not None:
            cur_node = cur_node.next
            elements.append(cur_node.data)
        return len(elements)


class TreeNode():
    def __init__(self, data=None, children=None):

        if children is None:
            children = []

        self.data = data
        self.children = children

    def add_child(self, child : 'TreeNode'):
        """
        Add a Child to the list of children.

        Parameters:
            child : (TreeNode obj)

        Appends the list with the new child.
        """
        if isinstance(child, TreeNode):
            self.children.append(child)
        else:
            raise TypeError('child must be an instance of TreeNode')

    def remove_child(self, child):
        """
        Removes given child from the list.
        """
        if child in self.children:
            self.children.remove(child)
        else:
            return f'The specified child is not a direct child of this node.'





class Tree():
    """Create a Tree Data Structure """

    def __init__(self):
        self.root = TreeNode()

    def __repr__(self):
        """
        Prints a representation of the tree.

        """

