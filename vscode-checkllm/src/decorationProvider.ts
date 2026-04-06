import * as vscode from 'vscode';
import { ResultLoader } from './resultLoader';

/** Green/red gutter icons next to test functions. */
export class DecorationProvider {
  private passDecoration: vscode.TextEditorDecorationType;
  private failDecoration: vscode.TextEditorDecorationType;
  private loader: ResultLoader;

  constructor(loader: ResultLoader) {
    this.loader = loader;

    this.passDecoration = vscode.window.createTextEditorDecorationType({
      gutterIconPath: undefined, // Will use colored dot via overview ruler
      overviewRulerColor: '#3fb950',
      overviewRulerLane: vscode.OverviewRulerLane.Left,
      before: {
        contentText: '\u25CF',
        color: '#3fb950',
        margin: '0 4px 0 0',
      },
    });

    this.failDecoration = vscode.window.createTextEditorDecorationType({
      overviewRulerColor: '#f85149',
      overviewRulerLane: vscode.OverviewRulerLane.Left,
      before: {
        contentText: '\u25CF',
        color: '#f85149',
        margin: '0 4px 0 0',
      },
    });
  }

  updateDecorations(editor: vscode.TextEditor): void {
    if (!editor.document.fileName.endsWith('.py')) return;

    const results = this.loader.getResultsForFile(editor.document.fileName);
    const passRanges: vscode.DecorationOptions[] = [];
    const failRanges: vscode.DecorationOptions[] = [];
    const text = editor.document.getText();

    for (const result of results) {
      const pattern = new RegExp(`^(\s*def\s+${this.escapeRegex(result.testName)}\s*\()`, 'm');
      const match = pattern.exec(text);
      if (!match) continue;

      const pos = editor.document.positionAt(match.index);
      const range = new vscode.Range(pos, pos);

      if (result.allPassed) {
        passRanges.push({ range });
      } else {
        failRanges.push({ range });
      }
    }

    editor.setDecorations(this.passDecoration, passRanges);
    editor.setDecorations(this.failDecoration, failRanges);
  }

  dispose(): void {
    this.passDecoration.dispose();
    this.failDecoration.dispose();
  }

  private escapeRegex(s: string): string {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}
