"""이미지 생성 백엔드 인터페이스."""

from typing import Protocol

from PIL import Image

from pixel2place.layout.schema import Layout


class LayoutRenderer(Protocol):
    """레이아웃에서 top-down 보드 이미지를 만든다.

    레이아웃을 입력으로 받는 렌더러만 구현한다. 지시문으로 이미지를 만드는 Cosmos 생성 클라이언트는
    이 인터페이스를 쓰지 않는다.
    """

    def render(self, layout: Layout) -> Image.Image: ...
