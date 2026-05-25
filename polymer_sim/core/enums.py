from enum import IntEnum


class ChannelBlock(IntEnum):
    LEFT_ADD = 0
    RIGHT_ADD = 1
    LEFT_SPLIT = 2
    RIGHT_SPLIT = 3
    OUTFLOW = 4


BLOCK_ORDER = (
    ChannelBlock.LEFT_ADD,
    ChannelBlock.RIGHT_ADD,
    ChannelBlock.LEFT_SPLIT,
    ChannelBlock.RIGHT_SPLIT,
    ChannelBlock.OUTFLOW,
)


BLOCK_NAMES = {
    ChannelBlock.LEFT_ADD: "LEFT_ADD",
    ChannelBlock.RIGHT_ADD: "RIGHT_ADD",
    ChannelBlock.LEFT_SPLIT: "LEFT_SPLIT",
    ChannelBlock.RIGHT_SPLIT: "RIGHT_SPLIT",
    ChannelBlock.OUTFLOW: "OUTFLOW",
}
