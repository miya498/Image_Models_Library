import collections
import torch
from torch.utils.tensorboard import SummaryWriter


TpuSummaryEntry = collections.namedtuple(
    "TpuSummaryEntry", "summary_fn name tensor reduce_fn"
)


class TpuSummaries:
    """
    PyTorch版のTPUサマリークラス。

    各サマリーエントリは、テンソルを指定された関数で集約し、
    TensorBoardに記録するために保存されます。
    """

    def __init__(self, log_dir, save_summary_steps=250):
        self._log_dir = log_dir
        self._entries = []
        self.record = True  # サマリーエントリを追加するかどうか
        self._save_summary_steps = save_summary_steps
        self.writer = SummaryWriter(log_dir=log_dir)

    def image(self, name, tensor, reduce_fn=None):
        """
        画像用のサマリーエントリを追加。
        """
        if not self.record:
            return
        if reduce_fn is None:
            reduce_fn = lambda x: x  # 画像ではデフォルトでそのまま
        self._entries.append(
            TpuSummaryEntry(self.writer.add_image, name, tensor, reduce_fn)
        )

    def scalar(self, name, tensor, reduce_fn=torch.mean):
        """
        スカラー用のサマリーエントリを追加。
        """
        if not self.record:
            return
        tensor = torch.tensor(tensor)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        self._entries.append(
            TpuSummaryEntry(self.writer.add_scalar, name, tensor, reduce_fn)
        )

    def write_summaries(self, global_step):
        """
        サマリーを実際にTensorBoardに書き込む。
        """
        for entry in self._entries:
            # reduce_fnで集約した値を取得
            value = entry.reduce_fn(entry.tensor)
            # TensorBoardに記録
            if entry.summary_fn == self.writer.add_image:
                # 画像の場合、`add_image`が要求するフォーマットに合わせる
                self.writer.add_image(entry.name, value, global_step=global_step)
            else:
                # スカラーやその他のデータ
                self.writer.add_scalar(entry.name, value.item(), global_step=global_step)

    def close(self):
        """
        TensorBoardのライターを閉じる。
        """
        self.writer.close()