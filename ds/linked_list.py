from __future__ import annotations
from typing import Generic, TypeVar, Iterable, Iterator, Callable, Optional

T = TypeVar("T")


class Node(Generic[T]):
    """
    A single node in a singly‑linked list.

    Parameters
    ----------
    value : T
        The data held by the node.
    next : Optional[Node[T]], default None
        Reference to the next node in the list, or ``None`` if this node
        is the tail.
    """
    __slots__ = ("value", "next")

    def __init__(self, value: T, next: Optional["Node[T]"] = None) -> None:
        self.value: T = value
        self.next: Optional["Node[T]"] = next

    def __repr__(self) -> str:  # pragma: no cover
        return f"Node({self.value!r})"


class LinkedList(Generic[T]):
    """
    A robust, user‑friendly implementation of a **singly‑linked list**.

    The class provides a Pythonic API with familiar dunder‑methods so instances
    behave much like built‑in mutable sequences (e.g. ``list``).

    Notes
    -----
    - All mutation methods run in *O(1)* time except :pymeth:`insert`,
      :pymeth:`__getitem__`, and :pymeth:`__setitem__`, which are *O(n)*.
    - Negative indices are supported wherever applicable.
    """

    __slots__ = ("_head", "_tail", "_length")

    def __init__(self, iterable: Optional[Iterable[T]] = None) -> None:
        """
        Create a new list, optionally populated from ``iterable``.

        Parameters
        ----------
        iterable : Iterable[T], optional
            An iterable whose items will be appended in order.
        """
        self._head: Optional[Node[T]] = None
        self._tail: Optional[Node[T]] = None
        self._length: int = 0
        if iterable is not None:
            self.extend(iterable)

    # --------------------------------------------------------------------- #
    # Properties and dunder‑methods
    # --------------------------------------------------------------------- #

    def __len__(self) -> int:  # pragma: no cover
        """Return ``len(self)``."""
        return self._length

    def __bool__(self) -> bool:  # pragma: no cover
        """``True`` if the list is non‑empty."""
        return self._length > 0

    def __iter__(self) -> Iterator[T]:
        """Yield items from head to tail."""
        current = self._head
        while current:
            yield current.value
            current = current.next

    def __repr__(self) -> str:  # pragma: no cover
        return f"LinkedList([{', '.join(repr(v) for v in self)}])"

    def _node_at(self, index: int) -> Node[T]:
        """
        Return the *Node* at ``index`` (supporting negative indices).

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        if not (-self._length <= index < self._length):
            raise IndexError("index out of range")

        if index < 0:
            index = self._length + index

        current = self._head
        for _ in range(index):
            current = current.next  # type: ignore[assignment]
        assert current is not None  # for mypy
        return current

    def __getitem__(self, index: int) -> T:
        """Return ``self[index]``."""
        return self._node_at(index).value

    def __setitem__(self, index: int, value: T) -> None:
        """Assign to ``self[index]``."""
        self._node_at(index).value = value

    def __contains__(self, value: object) -> bool:  # pragma: no cover
        return any(item == value for item in self)

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        if not isinstance(other, LinkedList):
            return NotImplemented
        return list(self) == list(other)

    # --------------------------------------------------------------------- #
    # Core mutation helpers
    # --------------------------------------------------------------------- #

    def _append_node(self, node: Node[T]) -> None:
        """Append *node* (assumed detached) to *self* in O(1)."""
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._length += 1

    # --------------------------------------------------------------------- #
    # Public mutation API
    # --------------------------------------------------------------------- #

    def append(self, value: T) -> None:
        """
        Append *value* to the tail of the list in O(1) time.
        """
        self._append_node(Node(value))

    def prepend(self, value: T) -> None:
        """
        Insert *value* at the head of the list in O(1) time.
        """
        node = Node(value, self._head)
        self._head = node
        if self._tail is None:
            self._tail = node
        self._length += 1

    def extend(self, iterable: Iterable[T]) -> None:
        """
        Append each item from *iterable*.

        This method is more efficient than repeated :pymeth:`append` calls
        because it reuses the internal *Node* objects when possible.
        """
        for item in iterable:
            self._append_node(Node(item))

    def insert(self, index: int, value: T) -> None:
        """
        Insert *value* before position *index* (like ``list.insert``).

        If *index* is equal to ``len(self)`` or greater, *value* is appended.
        If *index* is less than or equal to ``-len(self)``, *value* is prepended.
        """
        if index <= -self._length:
            self.prepend(value)
            return
        if index >= self._length:
            self.append(value)
            return

        if index < 0:
            index = self._length + index
        if index == 0:
            self.prepend(value)
            return

        prev_node = self._node_at(index - 1)
        new_node = Node(value, prev_node.next)
        prev_node.next = new_node
        self._length += 1

    def pop(self, index: int = -1) -> T:
        """
        Remove and return item at *index* (default the tail).

        Raises
        ------
        IndexError
            If the list is empty or *index* is out of range.
        """
        if self._length == 0:
            raise IndexError("pop from empty LinkedList")

        if index == 0 or index <= -self._length:
            # pop head
            assert self._head is not None
            value = self._head.value
            self._head = self._head.next
            if self._head is None:
                self._tail = None
            self._length -= 1
            return value

        if index < 0:
            index = self._length + index

        if not (0 <= index < self._length):
            raise IndexError("pop index out of range")

        prev = self._node_at(index - 1)
        assert prev.next is not None
        value = prev.next.value
        prev.next = prev.next.next
        if prev.next is None:
            self._tail = prev
        self._length -= 1
        return value

    def remove(self, value: T) -> None:
        """
        Remove the *first* occurrence of *value*.

        Raises
        ------
        ValueError
            If *value* is not present.
        """
        prev: Optional[Node[T]] = None
        current = self._head
        while current and current.value != value:
            prev, current = current, current.next

        if current is None:
            raise ValueError(f"{value!r} not in LinkedList")

        if prev is None:
            # Removing head
            self._head = current.next
            if self._head is None:
                self._tail = None
        else:
            prev.next = current.next
            if prev.next is None:
                self._tail = prev
        self._length -= 1

    def clear(self) -> None:
        """
        Remove all items in O(1) time by dropping references.
        """
        self._head = self._tail = None
        self._length = 0

    def reverse(self) -> None:
        """
        In‑place reversal of the list in O(n) time and O(1) space.
        """
        prev: Optional[Node[T]] = None
        current = self._head
        self._tail = self._head
        while current:
            nxt = current.next
            current.next = prev
            prev, current = current, nxt
        self._head = prev

    # --------------------------------------------------------------------- #
    # Convenience helpers
    # --------------------------------------------------------------------- #

    def find(self, predicate: Callable[[T], bool]) -> Optional[T]:
        """
        Return the first item for which ``predicate(item)`` is ``True`` or
        ``None`` if not found.
        """
        for item in self:
            if predicate(item):
                return item
        return None

    def to_list(self) -> list[T]:
        """
        Convert to a built‑in :class:`list` (shallow copy).
        """
        return list(self)

    @classmethod
    def from_iterable(cls, iterable: Iterable[T]) -> "LinkedList[T]":
        """
        Alternate constructor: ``LinkedList.from_iterable(range(10))``.
        """
        return cls(iterable)
