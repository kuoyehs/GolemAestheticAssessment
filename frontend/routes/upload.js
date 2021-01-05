var express = require('express');
var router = express.Router();
var { spawn, exec, execFile } = require('child_process')
var multer  = require('multer')
var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'python/')
    },
    filename: function (req, file, cb) {
        cb(null, 'demo.jpg')
  }
})
 
var upload = multer({ storage: storage })

router.get('/upload', function(req, res, next) {
  const fs = require('fs');
  fs.readFile('./public/files/output.txt', (e, data) => {
      if (e) throw e;
      res.render('upload', { title: 'VisualNext', data: data });
  });
});

router.post('/uploadimage', upload.single('imageupload'),function(req, res) {
  try {
    exec('cp ./python/demo.jpg ./public/images/', (error, stdout, stderr) => {
      if (error) {
        console.error('stderr', stderr);
        throw error;
      }
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader("Content-Type", "text/event-stream");
      res.send("Waiting...");
      console.log('stdout', 'Upload file success!');
      exec('/bin/sh ./python/run.sh', (error, stdout, stderr) => {
        if (error) {
          console.error('stderr', stderr);
          throw error;
        }
        exec('mv ./output.txt ./public/files/', (error, stdout, stderr) => {
          if (error) {
            console.error('stderr', stderr);
            throw error;
          }
          console.log('stdout', 'success!');
        });
        console.log('stdout', stdout);
      });
    });
  } catch (err) {
    console.log(err);
    next(err);
  }
});

module.exports = router;
