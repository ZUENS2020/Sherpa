import https from 'https';

const data = JSON.stringify({
  jobs: [
    {
      code_url: 'https://github.com/test/test',
      total_duration: 900,
      single_duration: 900,
      max_tokens: 0,
      unlimited_round_limit: 7200
    }
  ]
});

const options = {
  hostname: 'dev.zuens2020.work',
  port: 443,
  path: '/api/task',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': data.length
  }
};

const req = https.request(options, (res) => {
  let body = '';
  res.on('data', (chunk) => body += chunk);
  res.on('end', () => {
    console.log(`POST /api/task -> ${res.statusCode} ${body}`);
  });
});

req.on('error', (e) => {
  console.error(`Error: ${e.message}`);
});

req.write(data);
req.end();
