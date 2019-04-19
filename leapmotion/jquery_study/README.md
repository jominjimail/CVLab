# jquery

<img src="./images/1.png" width="20%">

- jquery version : 3.4.0
- 위치 : ./lib/jquery-3.4.0.min.js

## 라이브러리

자주 사용하는 로직을 재사용할 수 있도록 고안된 소프트웨어를 라이브러리라고 한다.

## jQuery

jQuery는 DOM을 내부에 감추고 보다 쉽게 웹페이지를 조작할 수 있도록 돕는 도구이다. 

- DOM : Document Object Model로 웹페이지를 자바스크립트로 제어하기 위한 객체 모델을 의미한다. window 객체의 document 프로퍼티를 통해서 사용할 수 있다. Window 객체가 창을 의미한다면 Document 객체는 윈도우에 로드된 문서를 의미한다고 할 수 있다.

- html 코드와 js 코드를 분리하고 싶다면 src 로 호출하면 된다. 단, jauery 라이브러리를 먼저 호출한뒤에 하자.

  ```html
  <!--test.html-->
  <script src="./lib/jquery-3.4.0.min.js"></script>
  <script type="text/javascript" src="jquery_basic2_compare.js"></script>
  
  <!--jquery_basic2_compare.js-->
  jQuery( document ).ready(function( $ ) {
  ...
      $('#jquery').on('click', function(event){
          alert('jQuery');
      })
  });
  ```





## example

```html
<ul id="demo">
    <li class="active">HTML</li>
    <li id="active">CSS</li>
    <li class="active">JavaScript</li>
</ul>
...
<!--javascript-->
var lis = document.getElementsByTagName('li');
for(var i=0; i&lt;lis.length; i++){
    lis[i].style.color='red';   
...
<!--jquery-->
$('li').css('color', 'red')
```

- li 의 색을 레드로 바꾼다.

```html
<!--javascript-->
var lis = document.getElementsByClassName('active');
for(var i=0; i &lt; lis.length; i++){
    lis[i].style.color='red';   
}
...
<!--jquery-->
$('.active').css('color', 'red')
```

- class 가 active 인것들의 색을 레드로 바꾼다.

```html
<!--javascript-->
var li = document.getElementById('active');
li.style.color='red';
li.style.textDecoration='underline';
...
<!--jquery-->
$('$active').css('color', 'red').css('textDecoration', 'underline');
```

- id가 active 인것들의 색을 레드로 바꾼다.