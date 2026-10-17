"""보드 이미지 판독기 인터페이스."""

from typing import Protocol

from PIL import Image

from pixel2place.layout.schema import Layout


class LayoutParser(Protocol):
    """top-down 보드 이미지에서 레이아웃을 읽는다.

    `GridImageParser`와 `CosmosReasoner`가 이 인터페이스를 구현한다.
    """

    def parse(self, image: Image.Image) -> Layout: ...
