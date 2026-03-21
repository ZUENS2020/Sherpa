import https from 'https';

const endpoints = [
  { path: '/api/tasks', method: 'POST' },
  { path: '/api/task', method: 'POST' },
  { path: '/api/fuzz', method: 'POST' },
  { path: '/api/submit', method: 'POST' },
  { path: '/api/run', method: 'POST' },
  { path: '/api/tasks', method: 'PUT' },
  { path: '/api/task', method: 'PUT' },
];

const data = JSON.stringify({
  repo: 'test',
  total_duration: 900,
  single_duration: 900,
  max_tokens: 0,
  unlimited_round_limit: 7200
});

async function testEndpoint({ path, method }) {
  return new Promise((resolve) => {
    const options = {
      hostname: 'dev.zuens2020.work',
      port: 443,
      path,
      method,
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length
      }
    };

    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        console.log(`${method} ${path} -> ${res.statusCode} ${body}`);
        resolve();
      });
    });

    req.on('error', (e) => {
      console.error(`${method} ${path} -> Error: ${e.message}`);
      resolve();
    });

    req.write(data);
    req.end();
  });
}

async function run() {
  for (const ep of endpoints) {
    await testEndpoint(ep);
  }
}

run();
