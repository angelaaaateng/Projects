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
