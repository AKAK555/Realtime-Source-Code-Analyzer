$(document).ready(function(){

    $("#indexRepo").click(function(){

        let repoUrl = $("#repo-url").val();

        if(repoUrl.trim() === "")
            return;

        $("#repo-status")
            .html(
                "<span class='loading'>Indexing repository...</span>"
            );

        $.ajax({

            type:"POST",

            url:"/chatbot",

            data:{
                question:repoUrl
            },

            success:function(result){

                $("#repo-status")
                    .html(
                        "<span class='success'>✓ Repository indexed successfully</span>"
                    );

            },

            error:function(){

                $("#repo-status")
                    .html(
                        "<span class='error'>Failed to index repository</span>"
                    );
            }
        });

    });

    $("#chat-form").submit(function(e){

        e.preventDefault();

        let message = $("#message").val();

        if(message.trim()==="")
            return;

        $("#chat-window").append(

            `
            <div class="message user">
                ${message}
            </div>
            `
        );

        $("#message").val("");

        $("#chat-window")
            .scrollTop(
                $("#chat-window")[0].scrollHeight
            );

        $.ajax({

            type:"POST",

            url:"/get",

            data:{
                msg:message
            },

            success:function(response){

                $("#chat-window").append(

                    `
                    <div class="message bot">
                        ${response}
                    </div>
                    `
                );

                $("#chat-window")
                    .scrollTop(
                        $("#chat-window")[0].scrollHeight
                    );
            }
        });

    });

});