const express = require('express');
const { exec } = require('child_process');
const bodyParser = require('body-parser');

const app = express();

app.use(bodyParser.json());

app.post('/compile', (req, res) => {
  const code = req.body.code;

  const command = `nvcc -o compiled_code ${code}`;

  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error compiling code: ${error}`);
      res.status(500).json({ error: 'Compilation failed' });
      return;
    }

    console.log(`Compilation successful: ${stdout}`);
    res.status(200).json({ message: 'Compilation successful', output: stdout });
  });
});

app.get('/', (req, res) => {
  res.status(200).json({ message: 'Server is up and running' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
