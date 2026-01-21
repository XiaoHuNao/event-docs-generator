# Event Docs Generator

一个零配置的 Python 工具，用于从 NeoForge/Forge 源码 Jar 和模组 Jar 中提取事件信息，并生成结构化的 Markdown 文档。

## 功能

- **零配置**：自动扫描当前目录及 `libs/`、`mods/` 子目录下的 Jar 文件。
- **详细文档**：
  - 生成事件类的继承关系树。
  - 提取 Javadoc 注释和注解（如 `@Cancelable`, `@HasResult`）。
  - 为每个模组生成单独的事件列表。
- **易于阅读**：生成的 Markdown 文档包含目录、代码片段和详细的类型信息。

## 如何使用

### 1. 环境准备

确保已安装 Python 3.8 或更高版本。

### 2. 获取 Jar 文件

为了生成完整的文档，你需要准备以下文件并放入脚本同级目录：

- **NeoForge/Forge 源码 Jar**：用于解析基础事件类。
  - 文件名通常以 `-sources.jar` 结尾。
  - 你可以在 Gradle 缓存中找到它们：
    - **NeoForge**: `~/.gradle/caches/modules-2/files-2.1/net.neoforged/neoforge/<版本>/<Hash>/neoforge-<版本>-sources.jar`
    - **Forge**: `~/.gradle/caches/forge_gradle/maven_downloader/net/minecraftforge/forge/<版本>/forge-<版本>-sources.jar`

- **模组 Jar**（可选）：如果你想生成特定模组的事件文档。
  - 将模组 Jar 文件放入脚本同级目录或 `mods/` 目录。

### 3. 运行脚本

在命令行中运行：

```bash
python event-docs-generator.py
```

### 4. 查看结果

运行完成后，文档将生成在 `docs/` 目录下：

- `docs/events.md`：包含所有扫描到的事件文档。
- `docs/<模组ID>.md`：各个模组的独立文档（如果有）。

## 目录结构示例

## 开发与贡献

欢迎提交 Issue 或 Pull Request 来改进这个工具！

## 许可证

MIT License
