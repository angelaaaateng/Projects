<html>
<head>

  <script>
    var source = new EventSource("/stream");
    source.onmessage = function(event) {
        document.getElementById("output").innerHTML += event.data + "<br/>"
    }
    </script>

</head>
</html>


<!doctype html>
<html>
   <body>
      <table border = 1>
         {% for key, value in result.iteritems() %}
            <tr>
               <th> {{ key }} </th>
               <td> {{ value }} </td>
            </tr>
         {% endfor %}
      </table>
   </body>
</html>
