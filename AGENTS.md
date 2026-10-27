# 博客整理规则

这是 Ray 的纯 Markdown 博客仓库，文章直接放在仓库根目录。不要引入 Hexo、Hugo、Jekyll 或其他静态站点生成器，也不要改成 `posts/` 目录。

## 仓库结构

```text
blog/
├── README.md       # 博客说明和文章索引
├── AGENTS.md       # 本文件，整理和发布规则
├── images/         # 文章图片，正文用相对路径引用
└── <文章标题>.md    # 文章正文，直接放根目录
```

## 文章文件

- 文件名使用文章标题，不加日期前缀，不加分类目录。
- 去掉标题末尾的句号、问号、感叹号等句末标点，标题中的 `pass@k`、`pass^k` 这类符号原样保留。
- 正文第一行写 `# 标题`，空一行后接正文。
- 不添加 YAML frontmatter。文章没有 `date`、`tags`、`category` 字段。
- 一篇文章一个文件，历史文章保持原位，不按主题重新分目录。
- 文章正文就是发布内容，不把 vault 里的写作大纲、备注、todo 或 frontmatter 带进来。

历史文章如果缺少一级标题，保留现状，不在整理索引时顺手改写正文。新文章必须有一级标题。

## 图片

- 图片统一放在 `images/`，不要散落在仓库根目录。
- 正文使用相对路径引用，例如 `![](images/example.png)`。
- 文件名尽量使用英文小写和连字符，避免空格和中文文件名带来的链接问题。
- 删除或重命名图片前，先搜索所有文章里的引用并同步修改。

## 索引

文章索引直接写在 `README.md` 里，不再单独创建 `INDEX.md`。

- 按发布时间从新到旧排列。
- 日期取文章在 Git 中的首次提交日期，格式为 `YYYY-MM-DD`。
- 标题优先使用文章的一级标题。没有一级标题时，使用去掉 `.md` 的文件名。
- 每篇文章一行，包含日期、标题链接和一句话摘要。
- 标题链接指向根目录下的文章文件。中文、空格和 `@`、`^` 等字符使用 URL 编码，避免 GitHub 渲染链接失败。
- 新文章发布时，在 README 的索引顶部加入新行，并同步更新总篇数。
- 修改文章标题时，同时修改文件名、正文一级标题和 README 中的链接。
- 不把 `README.md`、`AGENTS.md` 或图片文件列入文章索引。

## 发布流程

1. 先同步仓库，避免覆盖远端修改。

   ```bash
   git pull --rebase --autostash origin main
   ```

2. 从 vault 取正文，删除 frontmatter、写作大纲和备注，只保留文章内容。
3. 在仓库根目录创建 `<文章标题>.md`，补齐一级标题。
4. 把配图放进 `images/`，检查正文中的相对路径。
5. 更新 `README.md` 中的文章索引，确认日期和排序正确。
6. 检查 Markdown 链接、图片路径和 `git diff --check`。
7. 提交并推送。

   ```bash
   git add "<文章标题>.md" images README.md
   git commit -m "add article <文章标题>"
   git push origin main
   ```

8. 推送后验证本地和远端一致。

   ```bash
   git fetch origin -q
   test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" && echo OK
   ```

## 修改边界

- 不重写 Ray 的句子、不调整文章结构、不统一文风。
- 只做机械修正，例如专有名词拼写、明显笔误、坏链接和多余空格。
- 修改已有文章时，不删除历史内容，不把文章改写成摘要。
- 索引和规则文档使用 `docs:` 前缀提交；新增文章使用 `add article <标题>`。
- 任何整理动作都不应改变文章正文的含义。

## 与 vault 的关系

文章通常先写入 Obsidian vault 的 `AI-Agents/Articles/`，再同步到这个博客仓库。vault 里保留 frontmatter 和关联链接，博客里只保留正文。两边的内容可以对应，但格式不要互相复制。

## 检查清单

- [ ] 文章在根目录，文件名是标题
- [ ] 首行是一级标题
- [ ] 没有 frontmatter
- [ ] 图片在 `images/`，引用路径正确
- [ ] README 的索引已加入新文章并按日期倒序
- [ ] `git diff --check` 通过
- [ ] 已 push，且本地 `HEAD` 等于 `origin/main`
