"""
test_sequence_buffer.py

测试：

    SequenceBuffer

验证：

    update()
    is_ready()
    get_sequence()
    remove()
    clear()
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT)

from sequence.sequence_buffer import SequenceBuffer


def test_update_and_ready():
    """
    测试：

        update()
        is_ready()
    """

    print("=" * 60)
    print("TEST 1 : update() / is_ready()")
    print("=" * 60)

    buffer = SequenceBuffer(seq_len=30)

    pid = 1

    for i in range(29):

        feature = np.random.rand(
            34
        ).astype(np.float32)

        buffer.update(
            pid,
            feature
        )

    print(
        "Length:",
        len(buffer.buffers[pid])
    )

    print(
        "Ready:",
        buffer.is_ready(pid)
    )

    assert len(
        buffer.buffers[pid]
    ) == 29

    assert (
        buffer.is_ready(pid)
        is False
    )

    feature = np.random.rand(
        34
    ).astype(np.float32)

    buffer.update(
        pid,
        feature
    )

    print(
        "Length:",
        len(buffer.buffers[pid])
    )

    print(
        "Ready:",
        buffer.is_ready(pid)
    )

    assert len(
        buffer.buffers[pid]
    ) == 30

    assert (
        buffer.is_ready(pid)
        is True
    )

    print("PASS\n")


def test_get_sequence():
    """
    测试：

        get_sequence()
    """

    print("=" * 60)
    print("TEST 2 : get_sequence()")
    print("=" * 60)

    buffer = SequenceBuffer(
        seq_len=30
    )

    pid = 1

    for _ in range(30):

        feature = np.random.rand(
            34
        ).astype(np.float32)

        buffer.update(
            pid,
            feature
        )

    sequence = buffer.get_sequence(
        pid
    )

    print(
        "Sequence Shape:",
        sequence.shape
    )

    assert sequence.shape == (
        30,
        34
    )

    print("PASS\n")


def test_auto_pop():
    """
    测试：

        deque(maxlen)

    超过长度自动删除最旧帧
    """

    print("=" * 60)
    print("TEST 3 : auto pop")
    print("=" * 60)

    buffer = SequenceBuffer(
        seq_len=30
    )

    pid = 1

    for i in range(35):

        feature = np.ones(
            34,
            dtype=np.float32
        ) * i

        buffer.update(
            pid,
            feature
        )

    sequence = buffer.get_sequence(
        pid
    )

    print(
        "Shape:",
        sequence.shape
    )

    print(
        "First Frame Value:",
        sequence[0][0]
    )

    print(
        "Last Frame Value:",
        sequence[-1][0]
    )

    assert sequence.shape == (
        30,
        34
    )

    # 应保留5~34
    assert sequence[0][0] == 5

    assert sequence[-1][0] == 34

    print("PASS\n")


def test_remove():
    """
    测试：

        remove()
    """

    print("=" * 60)
    print("TEST 4 : remove()")
    print("=" * 60)

    buffer = SequenceBuffer()

    buffer.update(
        1,
        np.random.rand(34)
    )

    buffer.update(
        2,
        np.random.rand(34)
    )

    print(
        "Before:",
        list(buffer.buffers.keys())
    )

    buffer.remove(1)

    print(
        "After:",
        list(buffer.buffers.keys())
    )

    assert 1 not in buffer.buffers

    print("PASS\n")


def test_clear():
    """
    测试：

        clear()
    """

    print("=" * 60)
    print("TEST 5 : clear()")
    print("=" * 60)

    buffer = SequenceBuffer()

    for pid in range(5):

        buffer.update(
            pid,
            np.random.rand(34)
        )

    print(
        "Before:",
        len(buffer.buffers)
    )

    buffer.clear()

    print(
        "After:",
        len(buffer.buffers)
    )

    assert len(
        buffer.buffers
    ) == 0

    print("PASS\n")


def main():

    print()
    print("=" * 60)
    print("SequenceBuffer Unit Test")
    print("=" * 60)
    print()

    test_update_and_ready()

    test_get_sequence()

    test_auto_pop()

    test_remove()

    test_clear()

    print("=" * 60)
    print("ALL TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()