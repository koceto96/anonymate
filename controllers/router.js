const fs = require('fs');
const csv = require('csv');

exports.getPricing = (req, res) => {
  res.render('pricing', {
    title: 'pricing'
  });
};

exports.getExample = (req, res) => {
  var parser = csv.parse(function(err, output){
    res.render('csv', {
      title: 'Example',
      output
    });
  });
  fs.createReadStream('./people_test_data_csv.csv').pipe(parser);
};

exports.postExample = (req,res)=> {
  var parser = csv.parse(function(err, output){
    res.render('csvpost', {
      title: 'Example',
      output
    });
  });
  fs.createReadStream('./anondata.csv').pipe(parser);
}

exports.getHome = (req, res) => {
  res.render('home', {
    title: 'Home'
  });
};