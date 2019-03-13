/**
 * GET /signup
 * Signup page.
 */
exports.getSignup = (req, res) => {
  res.render('account/signup', {
    title: 'Create Account'
  });
};
exports.postSignup = (req, res, next) => {
  req.assert('email', 'Email is not valid').isEmail();
  req.assert('password', 'Password must be at least 4 characters long').len(4);
  req.assert('confirmPassword', 'Passwords do not match').equals(req.body.password);
  req.sanitize('email').normalizeEmail({ gmail_remove_dots: false });

  //create JSON object from signup form
  const user = {  name : req.body.name, 
                  sex : req.body.sex,
                  dob : req.body.dob,
                  education: req.body.education,
                  address: req.body.address,
                  phone : req.body.phone,
                  email : req.body.email
                }
  console.log(user)
  const errors = req.validationErrors();

  if (errors) {
    return res.redirect('/signup');
  }
  res.redirect('/');
};
