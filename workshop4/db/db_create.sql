create database db_logistics_chatbot;
create user dbadmin with encrypted password 'dbadmin321';
grant select on all tables in schema public TO dbadmin;
grant all privileges on database db_logistics_chatbot to dbadmin;