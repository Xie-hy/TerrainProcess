from torch import Tensor
from torch_geometric.nn import knn, radius


def knn_search(pos_x: Tensor, pos_y: Tensor, batch_x: Tensor, batch_y: Tensor, r: float, k: int) -> Tensor:
    r"""采用 knn 检索每个点的邻域，对 y 的每个点，搜索他在 x 中的邻域，x 和 y 可以是 相同的，也可以是不相同的。
    如果是不相同的点，通常 y 是降采样后的下一层级点。

    Args:
        pos_x (Tensor): 匹配点位置信息
            :math:`x\in\mathbb{R}^{N \times 3}`

        pos_y (Tensor): 搜索点位置信息
            :math:`y\in\mathbb{R}^{N' \times 3}`

        batch_x (Tensor): x 的 batch 索引
            :math:`b\in\mathbb{Z}^N\{0,0,0,\ldots,B-1\}`

        batch_y  (Tensor): y 的 batch 索引
            :math:`b\in\mathbb{Z}^N'\{0,0,0,\ldots,B-1\}`

        r (float): 搜索半径，knn 检索用不上

        k (int): K 邻域大小

    Returns:
        edge_index (Tensor): 返回相邻点的边信息，其中 I[0] 是 x 的索引，是不连续的; I[1] 是 y 的索引，是连续的
            :math:`I \in \mathbb{Z}^{2 \times E}, E=K \times N`

    :rtype: :class:`Tensor`
    """

    return knn(pos_x, pos_y, k, batch_x, batch_y).flip([0])


def radius_search(pos_x: Tensor, pos_y: Tensor, batch_x: Tensor, batch_y: Tensor, r: float, k: int) -> Tensor:
    r"""采用 ball 检索每个点的邻域，对 y 的每个点，搜索他在 x 中的邻域，x 和 y 可以是 相同的，也可以是不相同的。
    如果是不相同的点，通常 y 是降采样后的下一层级点。

    Args:
        pos_x (Tensor): 匹配点位置信息
            :math:`x\in\mathbb{R}^{N \times 3}`

        pos_y (Tensor): 搜索点位置信息
            :math:`y\in\mathbb{R}^{N' \times 3}`

        batch_x (Tensor): x 的 batch 索引
            :math:`b\in\mathbb{Z}^N\{0,0,0,\ldots,B-1\}`

        batch_y  (Tensor): y 的 batch 索引
            :math:`b\in\mathbb{Z}^N'\{0,0,0,\ldots,B-1\}`

        r (float): 搜索半径

        k (int): K 邻域大小，在这里用来确定最大检索数量，设置为 2 * K

    Returns:
        edge_index (Tensor): 返回相邻点的边信息，其中 I[0] 是 x 的索引，是不连续的; I[1] 是 y 的索引，是连续的
            :math:`I \in \mathbb{Z}^{2 \times E}, E=K \times N`

    :rtype: :class:`Tensor`
    """
    return radius(pos_x, pos_y, r, batch_x, batch_y, max_num_neighbors=2 * k)


def create_searcher(method: str = 'radius'):
    r"""搜索邻域接口函数

        返回的邻域是一个二维数组，记录点之间的邻接关系图。对 y 的每个点，搜索他在 x 中的邻域，x 和 y 可以是 相同的，也可以是不相同的。
        如果是不相同的点，通常 y 是降采样后的下一层级点。
        其中 I[0] 是 x 的索引，是不连续的; I[1] 是 y 的索引，是连续的。

        :math:`I \in \{\mathbb{Z}^{E}, \mathbb{Z}^{E}\}, E=K \times N`

    Args:
        method (str): 搜索方法，可以是 radius 或 knn

    Returns:
        func: 返回接口函数
            (pos_x: Tensor, pos_y: Tensor, batch_x: Tensor, batch_y: Tensor, r: float, k: int) -> Tensor
    """
    if method == 'radius':
        return radius_search
    else:
        return knn_search
