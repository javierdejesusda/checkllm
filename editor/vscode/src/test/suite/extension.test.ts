/**
 * Smoke test: verify the extension activates and registers commands.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';

suite('checkllm extension', () => {
  test('activates and registers commands', async () => {
    const ext = vscode.extensions.getExtension('checkllm.checkllm-vscode');
    assert.ok(ext, 'extension not found');
    await ext!.activate();

    const commands = await vscode.commands.getCommands(true);
    assert.ok(commands.includes('checkllm.runTest'));
    assert.ok(commands.includes('checkllm.runSuite'));
    assert.ok(commands.includes('checkllm.dashboard'));
  });
});
