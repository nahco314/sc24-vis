# sc24-vis

SuperCon2024 の本選問題をビジュアライズするツールです。

## インストール

```commandline
pip install sc24-vis@git+https://github.com/nahco314/sc24-vis
```

## 使い方

```commandline
# 盤面全体のビジュアライズ
sc24-vis vis <入力ファイル> <解答プログラムの出力の入ったファイル>

# ズームしたビジュアライズ
sc24-vis vis-zoom <入力ファイル> <解答プログラムの出力の入ったファイル>
```

ffmpegがシステムにインストールされている場合、--generate-mp4フラグでmp4ファイルを生成します。

細かいオプションがいくつか存在するので、詳しくは `--help` を付けて実行するなどで確認してください。

## 謝辞

シミュレーション部分や入力読み込み部分のコードは、[hiikunz](https://github.com/hiikunz) から提供していただきました。ありがとうございます。
