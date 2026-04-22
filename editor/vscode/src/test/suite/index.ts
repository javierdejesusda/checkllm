/**
 * Mocha suite bootstrapper for the VS Code extension tests.
 */

import * as path from 'path';

export async function run(): Promise<void> {
  // Use a dynamic require so Mocha is only resolved when a real VS
  // Code host is driving the tests.
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const Mocha = require('mocha');
  const mocha = new Mocha({ui: 'tdd', color: true});
  mocha.addFile(path.resolve(__dirname, 'extension.test.js'));

  await new Promise<void>((resolve, reject) => {
    try {
      mocha.run((failures: number) => {
        if (failures > 0) {
          reject(new Error(`${failures} tests failed.`));
        } else {
          resolve();
        }
      });
    } catch (err) {
      reject(err);
    }
  });
}
