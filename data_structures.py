class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None


class LinkedList:
    """
    Create a Linked List.
    """

    def __init__(self):
        self.head = Node()

    def append(self, data):
        # Creating a new node with the data
        new_node = Node(data)

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

if __name__ == '__main__':
