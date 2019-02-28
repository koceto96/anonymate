const fs = require('fs');
const csv = require('csv');


exports.getCsv = (req, res) => {

  var parser = csv.parse(function(err, output){
    res.render('csv', {
      title: 'Home',
      output
    });
  });

  fs.createReadStream('./people_test_data_csv.csv').pipe(parser);
};

exports.postCsv = (req,res)=> {
  var parser = csv.parse(function(err, output){
    res.render('csvpost', {
      title: 'Home',
      output
    });
  });
  fs.createReadStream('./anon_data.csv').pipe(parser);
}