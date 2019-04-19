jQuery( document ).ready(function( $ ) {
    var target = document.getElementById('pure');
    if(target.addEventListener){
        target.addEventListener('click', function(event){
            alert('pure');
        });
    } else {
        target.attachEvent('onclick', function(event){
            alert('pure');
        });
    }

    $('#jquery').on('click', function(event){
        alert('jQuery');
    })
    
  });