/* eslint-env jquery, browser */
$(document).ready(() => {

  // Place JavaScript code here...
  $('#load').click(function (){
    console.log("test")
    $(this).hide()
    $(this).parent().append('<button class="btn btn-success float-right" type="button" disabled><span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...</button>')
  })
});
